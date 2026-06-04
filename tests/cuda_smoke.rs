#[cfg(feature = "cuda")]
mod cuda_smoke {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use quantale_semiring_v2::{
        AsyncUploadQueue, DeviceSlotRegistry, JitCache, JitChain, OperatorRegistry,
        TensorQuantaleWorld, load_operator_registry,
    };

    #[test]
    fn cuda_feature_runtime_smoke_test() -> Result<(), Box<dyn std::error::Error>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };

        let registry = load_operator_registry("assets/operators.generated.json")?;
        let chain = JitChain {
            operators: vec!["Analysis::Return1".to_string()],
            inputs: vec!["market.price".to_string(), "market.open".to_string()],
            outputs: vec!["analysis.return".to_string()],
            internals: vec![],
        };

        let func = JitCache::new()
            .get_or_compile(&device, &chain, &registry as &OperatorRegistry)
            .map_err(|e| format!("jit compile failed: {e}"))?;
        let price = device.htod_copy(vec![110.0f32, 120.0, 90.0, 105.0])?;
        let open = device.htod_copy(vec![100.0f32, 100.0, 100.0, 100.0])?;
        let mut out = device.htod_copy(vec![0.0f32; 4])?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, (&price, &open, &mut out, 4_i32))? };
        let actual = device.dtoh_sync_copy(&out)?;
        let expected = [0.1f32, 0.2, -0.1, 0.05];
        for (idx, (&a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 1e-5, "idx={idx} actual={a} expected={e}");
        }
        Ok(())
    }

    #[test]
    fn gpu_dispatch_region_uses_device_slot_registry() -> Result<(), Box<dyn std::error::Error>> {
        let mut world = match TensorQuantaleWorld::empty() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("skip: TensorQuantaleWorld::empty: {e}");
                return Ok(());
            }
        };
        let device = world.device().clone();

        let mut slots = DeviceSlotRegistry::new();
        slots.insert("math.a", device.htod_copy(vec![1.0f32, 2.0, 3.0, 4.0])?);
        slots.insert("math.b", device.htod_copy(vec![10.0f32, 20.0, 30.0, 40.0])?);
        slots.insert("math.add_out", device.htod_copy(vec![0.0f32; 4])?);

        world.gpu_dispatch_region_with_slots(&slots, 0, 0, 1, 0)?;

        let actual = device.dtoh_sync_copy(slots.get("math.add_out").unwrap())?;
        assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn async_upload_queue_flushes_pinned_host_data() -> Result<(), Box<dyn std::error::Error>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: no cuda device: {e}");
                return Ok(());
            }
        };

        let mut queue = AsyncUploadQueue::new();
        let mut slots = DeviceSlotRegistry::new();
        queue.stage("math.a", &[1.0, 2.0, 3.0, 4.0])?;
        queue.flush(&mut slots, &device)?;
        assert_eq!(queue.pending(), 0);
        assert_eq!(queue.in_flight(), 1);
        queue.synchronize()?;
        assert_eq!(queue.in_flight(), 0);

        let actual = device.dtoh_sync_copy(slots.get("math.a").unwrap())?;
        assert_eq!(actual, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_smoke_skipped_without_cuda_feature() {}
