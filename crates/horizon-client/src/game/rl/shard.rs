//! Shards (docs/rl/design.md, process model B): `--rl-client` adds RL clients to this headless
//! process, beside its own. Each is a `Game` of its own, joined to the server under its own name
//! with its own socket and shared memory, and run by the same headless loop on a thread of its
//! own, the way splitscreen's players are `Game`s in P1's process. They share what is read-only:
//! `Resources`, the packs, and with `--rl-image` the wgpu device (a renderer each, made on the
//! main thread over the first one's device and pipelines).
//!
//! The clients live and die together: when one's loop ends (a quit, an error, a stop signal) the
//! others end theirs, and the supervisor restarts the process with all of them. Python sees each
//! as a reset (`python/horizon_rl/fleet.py`, `clients_per_process`).
//!
//! Process-wide state is what they share: thread-locals are per client (each runs on its own
//! thread); the statics are caches of read-only data, mod hooks and the process's settings, the
//! same for every client of one command line.

use super::super::super::{Game, UiResources, loading_window::start_game, split::set_options_path};
use super::RlOptions;
use horizon_client::{audio::Audio, gfx::Renderer};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Instant,
};

/// `--rl-client name=N,socket=S[,shm=M][,target=T][,dir=D]`: one more client in this process.
#[derive(Clone, Debug, PartialEq)]
pub(in super::super::super) struct Peer {
    pub name: String,
    pub socket: PathBuf,
    pub shm: Option<PathBuf>,
    pub target: Option<String>,
    /// Its own folder (its `options.txt`); default: the process's.
    pub dir: Option<PathBuf>,
}

impl Peer {
    pub(in super::super::super) fn parse(spec: &str) -> Result<Self, String> {
        let (mut name, mut socket) = (None, None);
        let mut peer = Peer {
            name: String::new(),
            socket: PathBuf::new(),
            shm: None,
            target: None,
            dir: None,
        };
        for part in spec.split(',') {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("--rl-client {spec}: {part:?} is not key=value"))?;
            if value.is_empty() {
                return Err(format!("--rl-client {spec}: {key} is empty"));
            }
            match key {
                "name" => name = Some(value.to_owned()),
                "socket" => socket = Some(PathBuf::from(value)),
                "shm" => peer.shm = Some(value.into()),
                "target" => peer.target = Some(value.to_owned()),
                "dir" => peer.dir = Some(value.into()),
                _ => return Err(format!("--rl-client {spec}: unknown key {key}")),
            }
        }
        peer.name = name.ok_or_else(|| format!("--rl-client {spec}: no name"))?;
        peer.socket = socket.ok_or_else(|| format!("--rl-client {spec}: no socket"))?;
        Ok(peer)
    }
}

/// Set once any client of the process has stopped: the others stop too.
static ENDED: AtomicBool = AtomicBool::new(false);

/// Whether a client of this process has stopped (always false without `--rl-client`).
pub(in super::super::super) fn ended() -> bool {
    ENDED.load(Ordering::Relaxed)
}

/// The other clients' threads. Dropped when the first client's loop is over: it tells them to
/// stop and waits for them to close.
#[derive(Default)]
pub(in super::super::super) struct Shard {
    threads: Vec<JoinHandle<()>>,
}

impl Drop for Shard {
    fn drop(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        ENDED.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// A `Game` is ~67 KB and its loop runs deep: the main thread's stack size.
const STACK: usize = 8 << 20;

impl Game<'_> {
    /// Starts the `--rl-client`s, each on its thread; none without the switch.
    pub(in super::super::super) fn rl_shard(&mut self) -> Shard {
        let mut shard = Shard::default();
        let Some(peers) = self
            .options
            .rl
            .as_mut()
            .map(|r| std::mem::take(&mut r.peers))
        else {
            return shard;
        };
        if peers.is_empty() {
            return shard;
        }
        // The clients borrow the resources for the process's lifetime: one copy for all.
        let res: &'static Arc<super::super::super::Resources> =
            Box::leak(Box::new(self.res.clone()));
        let vanilla_root = self.vanilla_root();
        for (i, peer) in peers.into_iter().enumerate() {
            match self.rl_peer(res, &vanilla_root, i + 1, peer) {
                Ok(t) => shard.threads.push(t),
                Err(e) => {
                    log::error!("RL shard: client {} not started: {e}", i + 1);
                    ENDED.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
        log::info!(
            "RL shard: {} clients in this process",
            shard.threads.len() + 1
        );
        shard
    }

    /// One more client: its options (this one's with its name, socket, shared memory and
    /// target), its UI resources and, rendering, its renderer on this device; then its thread.
    fn rl_peer(
        &self,
        res: &'static Arc<super::super::super::Resources>,
        vanilla_root: &std::path::Path,
        seat: usize,
        peer: Peer,
    ) -> horizon_client::Result<JoinHandle<()>> {
        let mut options = self.options.clone();
        options.player_name = Some(peer.name.clone());
        options.control = None;
        let rl = options.rl.get_or_insert_with(RlOptions::default);
        rl.socket = peer.socket;
        rl.shm = peer.shm;
        rl.target = peer.target;
        // Without a renderer it draws nothing: the font's metrics are all it reads of the GUI.
        let mut gui = match self.renderer.is_some() {
            true => UiResources::load(&options.resources, vanilla_root)?,
            false => UiResources::server(&options.resources)?,
        };
        gui.no_title_panorama();
        let renderer = match self.renderer.is_some() {
            true => {
                // Built from the same packs as the first client's: its read-only textures
                // serve this one too; the atlas only without the HUD, where no text writes
                // glyphs into it (gfx/share.rs).
                let hud = options.rl.as_ref().is_some_and(|rl| rl.hud);
                let _siblings = horizon_client::gfx::share::Siblings::hold(!hud);
                let mut r =
                    self.renderer_for(None::<Arc<winit::window::Window>>, &self.pack, &gui)?;
                let [w, h] = options.window_size;
                r.resize(w, h);
                Some(r)
            }
            false => None,
        };
        let job = Job {
            res,
            pack: self.pack.clone(),
            gui,
            settings: self.settings.clone(),
            audio: Audio::silent(&options.resources),
            options_file: peer.dir.map(|d| d.join("options.txt")),
            options,
            renderer,
            vanilla_root: vanilla_root.to_owned(),
        };
        Ok(std::thread::Builder::new()
            .name(format!("RL client {seat}"))
            .stack_size(STACK)
            .spawn(move || job.run(seat))?)
    }
}

/// What a client's thread starts its game from.
struct Job {
    res: &'static Arc<super::super::super::Resources>,
    pack: super::super::super::PackResources,
    gui: UiResources,
    settings: horizon_client::settings::Settings,
    audio: Audio,
    options: super::super::super::launch_options::Options,
    options_file: Option<PathBuf>,
    renderer: Option<Renderer>,
    vanilla_root: PathBuf,
}

impl Job {
    fn run(self, seat: usize) {
        // Its options file is its own (a thread-local, as splitscreen's players have).
        set_options_path(self.options_file);
        let started = start_game(
            self.res,
            self.pack,
            self.gui,
            self.options,
            None,
            self.audio,
            self.vanilla_root,
            Instant::now(),
            None,
            Some(self.settings),
            None,
            None,
        );
        match started {
            Ok(Some(game)) => {
                let mut game = Box::new(game);
                if let Some(renderer) = self.renderer {
                    game.renderer = Some(renderer);
                    game.world_renderer_ready = true;
                    game.apply_settings();
                }
                game.run_headless();
                if let Some(e) = game.error.take() {
                    log::error!("RL client {seat}: {e}");
                }
            }
            Ok(None) => {}
            Err(e) => log::error!("RL client {seat} did not start: {e}"),
        }
        // One client gone: the process's others follow, and the supervisor restarts them all.
        ENDED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_client_spec_names_its_socket_and_may_name_the_rest() {
        let p =
            Peer::parse("name=Agent_1,socket=/tmp/a/rl.sock,target=Agent_0,dir=/tmp/a").unwrap();
        assert_eq!(p.name, "Agent_1");
        assert_eq!(p.socket, PathBuf::from("/tmp/a/rl.sock"));
        assert_eq!(p.target.as_deref(), Some("Agent_0"));
        assert_eq!(p.dir, Some(PathBuf::from("/tmp/a")));
        assert_eq!(p.shm, None);
        assert!(Peer::parse("socket=/tmp/a").is_err(), "no name");
        assert!(Peer::parse("name=A").is_err(), "no socket");
        assert!(Peer::parse("name=A,socket=/s,colour=red").is_err());
        assert!(Peer::parse("name=A,socket=/s,target=").is_err());
    }
}
