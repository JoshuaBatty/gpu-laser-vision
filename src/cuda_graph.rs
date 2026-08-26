//! Thin interop layer between cuda-oxide and cuTile CUDA graphs.

use std::{ffi::c_void, sync::Arc};

use anyhow::{Context, Result};
use cuda_core::{CudaContext, CudaStream};
use cutile_cuda_async::{cuda_graph::CudaGraph, device_operation::DeviceOp, error::DeviceError};
use cutile_cuda_core::{Device, Stream};

/// A captured cuTile graph backed by a cuda-oxide context and stream.
pub(crate) struct CapturedCudaGraph {
    graph: CudaGraph<()>,
    stream: Arc<Stream>,
    _device: Arc<Device>,
}

impl CapturedCudaGraph {
    /// Borrows cuda-oxide handles while retaining their owning `Arc`s.
    pub(crate) fn capture(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        capture: impl FnOnce() -> Result<(), DeviceError>,
    ) -> Result<Self> {
        // SAFETY: cuTile retains the context owner and never destroys the borrowed handle.
        let device = unsafe {
            Device::borrow_with_owner(
                context.cu_ctx().cast::<c_void>(),
                context.cu_device(),
                context.ordinal(),
                context,
            )
        };
        // SAFETY: cuTile retains the stream owner and never destroys the borrowed handle.
        let stream = unsafe {
            Stream::borrow_with_owner(stream.cu_stream().cast::<c_void>(), &device, stream)
        };
        let graph = CudaGraph::scope(&stream, |_| capture()).context("capturing CUDA graph")?;

        Ok(Self {
            graph,
            stream,
            _device: device,
        })
    }

    /// Launches the graph and waits for its root stream to complete.
    pub(crate) fn launch(&self) -> Result<()> {
        self.graph
            .launch()
            .sync_on(&self.stream)
            .context("launching CUDA graph")?;
        Ok(())
    }
}
