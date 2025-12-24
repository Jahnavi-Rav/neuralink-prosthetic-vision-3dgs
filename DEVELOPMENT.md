# Development Roadmap

## Phase 1: Setup & Baseline (Weeks 1-5)
- [x] Project structure created
- [x] Environment setup
- [ ] Replica dataset downloaded
- [ ] Baseline 3DGS training working
- [ ] Achieve 30+ PSNR on test views
- [ ] Optimize to 30 FPS on GPU

## Phase 2: Semantic Enhancement (Weeks 6-8)
- [ ] SegFormer integration
- [ ] Priority region detection (faces, people)
- [ ] Real-time segmentation pipeline
- [ ] Semantic-aware rendering

## Phase 3: Neural Encoding (Weeks 7-9)
- [ ] Cortical encoder implementation
- [ ] Phosphene simulator
- [ ] Retinotopic mapping
- [ ] Hardware constraint modeling

## Phase 4: Eye Tracking (Weeks 9-10)
- [ ] Gaze simulation
- [ ] Foveated rendering (3 levels)
- [ ] Dynamic electrode allocation
- [ ] Latency optimization (<33ms)

## Phase 5: Paper & Demo (Weeks 11-12)
- [ ] Benchmarking complete
- [ ] Demo video recorded
- [ ] Paper draft (CVPR format)
- [ ] GitHub polished
- [ ] Submit to arXiv

## Key Metrics to Track
- Training PSNR/SSIM: Target >30 PSNR
- Inference FPS: Target >30 FPS
- End-to-end latency: Target <33ms
- Phosphene perception quality: Target >70% object recognition
- Memory footprint: Target <8GB
