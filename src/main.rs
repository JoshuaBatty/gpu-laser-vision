mod edge_detection;
mod kernels;

use anyhow::Result;

fn main() -> Result<()> {
    edge_detection::process("assets/test_tiles.png")?.save("assets")
}
