//! Thin interop layer between cuda-oxide and cuTile CUDA graphs.

use std::{ffi::c_void, sync::Arc};

use anyhow::{Context, Result};
use cuda_core::{CudaContext, CudaStream};
use cutile_cuda_async::{cuda_graph::CudaGraph, device_operation::DeviceOp, error::DeviceError};
use cutile_cuda_core::{Device, Stream};

/// A captured cuTile graph and the resources referenced by its nodes.
///
/// Field order is intentional: the graph and borrowed stream are destroyed
/// before the resource payload that owns their modules and device pointers.
pub(crate) struct CapturedCudaGraph<R> {
    graph: CudaGraph<()>,
    stream: Arc<Stream>,
    resources: R,
}

impl<R> CapturedCudaGraph<R> {
    /// Captures work against `resources` and retains them for the graph's lifetime.
    pub(crate) fn capture(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        mut resources: R,
        capture: impl FnOnce(&mut R) -> Result<(), DeviceError>,
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
        let graph = CudaGraph::scope(&stream, |_| capture(&mut resources))
            .context("capturing CUDA graph")?;

        Ok(Self {
            graph,
            stream,
            resources,
        })
    }

    /// Returns the resources owned by the captured graph.
    pub(crate) fn resources_mut(&mut self) -> &mut R {
        &mut self.resources
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
