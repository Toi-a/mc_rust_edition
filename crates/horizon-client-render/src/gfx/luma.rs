//! Grayscale captures of the offscreen picture for RL image observations (docs/rl/design.md,
//! "Images"): the frame is copied into a buffer, a compute pass turns it into one luma byte per
//! pixel, and the bytes come back through a mapped staging buffer, with no PNG and no colour
//! conversion on the CPU.
//!
//! It reads the picture the display would show (the offscreen target, after the tone curve and
//! the HUD), so with HDR on it would get HDR output's encoding: the RL client keeps HDR off
//! (docs/hdr-passes.md, rule 8: a capture maps the picture to SDR).

use super::Renderer;

/// The buffers and pipeline of one capture size, kept from frame to frame.
pub struct LumaCapture {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    row_bytes: u32,
    frame: wgpu::Buffer,
    luma: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

impl LumaCapture {
    /// For a `width` x `height` picture in `format`; `width` must be a multiple of 4 (four
    /// pixels share one u32 of the output), and the format 8-bit RGBA or BGRA.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        use wgpu::TextureFormat as F;
        let bgra = match format {
            F::Bgra8Unorm | F::Bgra8UnormSrgb => 1u32,
            F::Rgba8Unorm | F::Rgba8UnormSrgb => 0,
            other => return Err(format!("no luma capture of a {other:?} frame")),
        };
        if width == 0 || height == 0 || !width.is_multiple_of(4) {
            return Err(format!(
                "a luma capture needs a width that is a multiple of 4 ({width}x{height})"
            ));
        }
        let row_bytes = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = |label, size: u32, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(size),
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let frame = buffer("Luma frame", row_bytes * height, U::STORAGE | U::COPY_DST);
        let luma = buffer("Luma bytes", width * height, U::STORAGE | U::COPY_SRC);
        let staging = buffer("Luma staging", width * height, U::MAP_READ | U::COPY_DST);
        let params: Vec<u8> = [width, height, row_bytes / 4, bgra]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let params_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Luma params"),
                contents: &params,
                usage: U::UNIFORM,
            },
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Luma capture"),
            source: wgpu::ShaderSource::Wgsl(include_str!("luma.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Luma capture"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Luma capture"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: luma.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            width,
            height,
            format,
            row_bytes,
            frame,
            luma,
            staging,
            bind_group,
            pipeline,
        })
    }

    /// Whether this capture fits a `width` x `height` frame in `format`.
    pub fn fits(&self, width: u32, height: u32, format: wgpu::TextureFormat) -> bool {
        (self.width, self.height, self.format) == (width, height, format)
    }

    /// Records the copy of `texture` and the pass that turns it into luma bytes.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.frame,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.row_bytes),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Luma capture"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((self.width / 4).div_ceil(8), self.height.div_ceil(8), 1);
        }
        let size = u64::from(self.width * self.height);
        encoder.copy_buffer_to_buffer(&self.luma, 0, &self.staging, 0, size);
    }

    /// Waits for `submission` and copies the luma bytes (`width * height`, rows top first)
    /// into `out`.
    pub fn read(
        &self,
        device: &wgpu::Device,
        submission: wgpu::SubmissionIndex,
        out: &mut [u8],
    ) -> Result<(), String> {
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let result = match slice.get_mapped_range() {
            Ok(bytes) if bytes.len() == out.len() => {
                out.copy_from_slice(&bytes);
                Ok(())
            }
            Ok(bytes) => Err(format!(
                "luma capture of {} bytes for {}",
                bytes.len(),
                out.len()
            )),
            Err(e) => Err(e.to_string()),
        };
        self.staging.unmap();
        result
    }
}

impl Renderer {
    /// The offscreen picture's size, `None` with a window surface.
    pub fn offscreen_size(&self) -> Option<[u32; 2]> {
        self.offscreen.as_ref().map(|t| [t.width(), t.height()])
    }

    /// The last frame drawn offscreen as luma bytes in `out` (`width * height` of the
    /// offscreen size, rows top first). `capture` keeps the buffers between calls; it is made
    /// again when the size or format changes. Waits for the GPU.
    pub fn capture_luma(
        &self,
        capture: &mut Option<LumaCapture>,
        out: &mut [u8],
    ) -> Result<(), String> {
        let texture = self.offscreen.as_ref().ok_or("no offscreen frame")?;
        let (w, h, format) = (texture.width(), texture.height(), texture.format());
        if !capture.as_ref().is_some_and(|c| c.fits(w, h, format)) {
            *capture = Some(LumaCapture::new(&self.device, w, h, format)?);
        }
        let capture = capture.as_ref().expect("made above");
        let mut encoder = self.device.create_command_encoder(&Default::default());
        capture.record(&mut encoder, texture);
        // The queue waits for the frame on the submit thread first, so this comes after it.
        let submission = self.queue.submit([encoder.finish()]);
        capture.read(&self.device, submission, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    /// A 2-row picture with known pixels comes back as their Rec. 601 luma, in row order and
    /// without the copy's row padding, in both byte orders.
    #[test]
    fn known_pixels_come_back_as_their_luma() {
        let Some((device, queue)) = device() else {
            return;
        };
        let (w, h) = (8u32, 2u32);
        // RGBA: white, black, red, green, blue, grey 128, yellow, a dark blue-ish one.
        let rgba: [[u8; 4]; 8] = [
            [255, 255, 255, 255],
            [0, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [128, 128, 128, 255],
            [255, 255, 0, 255],
            [10, 20, 200, 255],
        ];
        // Independent expected values: round(0.299 R + 0.587 G + 0.114 B).
        let expected_row: Vec<u8> = rgba
            .iter()
            .map(|p| {
                (0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]))
                    .round() as u8
            })
            .collect();
        assert_eq!(&expected_row[..5], &[255, 0, 76, 150, 29]);
        for format in [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ] {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Row 0 as listed, row 1 reversed.
            let mut texels = Vec::new();
            for row in 0..h {
                let mut pixels = rgba.to_vec();
                if row == 1 {
                    pixels.reverse();
                }
                for p in pixels {
                    let p = if format == wgpu::TextureFormat::Bgra8Unorm {
                        [p[2], p[1], p[0], p[3]]
                    } else {
                        p
                    };
                    texels.extend_from_slice(&p);
                }
            }
            queue.write_texture(
                texture.as_image_copy(),
                &texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                texture.size(),
            );
            let capture = LumaCapture::new(&device, w, h, format).expect("a capture");
            let mut encoder = device.create_command_encoder(&Default::default());
            capture.record(&mut encoder, &texture);
            let submission = queue.submit([encoder.finish()]);
            let mut out = vec![0u8; (w * h) as usize];
            capture.read(&device, submission, &mut out).expect("read");
            let mut reversed = expected_row.clone();
            reversed.reverse();
            assert_eq!(&out[..8], &expected_row[..], "{format:?} row 0");
            assert_eq!(&out[8..], &reversed[..], "{format:?} row 1");
        }
    }

    #[test]
    fn widths_that_are_not_a_multiple_of_four_are_refused() {
        let Some((device, _)) = device() else {
            return;
        };
        assert!(LumaCapture::new(&device, 6, 4, wgpu::TextureFormat::Rgba8Unorm).is_err());
        assert!(LumaCapture::new(&device, 8, 4, wgpu::TextureFormat::Rgba16Float).is_err());
    }
}
