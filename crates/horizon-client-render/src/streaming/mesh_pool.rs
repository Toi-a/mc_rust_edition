//! The meshing workers' pool of cleared mesh buffers.

use crate::mesher::SectionMesh;

/// Cleared meshes kept for the next mesh jobs, up to [`MeshPool::limit`] bytes of capacity.
#[derive(Default)]
pub(super) struct MeshPool {
    pub(super) meshes: Vec<SectionMesh>,
    pub(super) bytes: usize,
}
impl MeshPool {
    pub(super) const BYTES: usize = 16 << 20;
    pub(super) const COUNT: usize = 128;
    /// The capacity kept: [`Self::BYTES`], or `HORIZON_MESH_POOL_BYTES` when that is less (an
    /// RL image client meshes a few columns around a player: docs/rl/design.md, "M5
    /// measurements").
    fn limit() -> usize {
        static LIMIT: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            std::env::var("HORIZON_MESH_POOL_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .map_or(MeshPool::BYTES, |bytes: usize| bytes.min(MeshPool::BYTES))
        });
        *LIMIT
    }
    pub(super) fn take(&mut self) -> SectionMesh {
        let mesh = self.meshes.pop().unwrap_or_default();
        self.bytes -= mesh.capacity_bytes();
        mesh
    }
    /// A rejected mesh is dropped by the caller, after releasing the lock.
    pub(super) fn put(&mut self, mut mesh: SectionMesh) -> Option<SectionMesh> {
        let bytes = mesh.capacity_bytes();
        let room = Self::limit().saturating_sub(self.bytes);
        if bytes == 0 || self.meshes.len() == Self::COUNT || bytes > room {
            return Some(mesh);
        }
        mesh.clear();
        self.bytes += bytes;
        self.meshes.push(mesh);
        None
    }
}
