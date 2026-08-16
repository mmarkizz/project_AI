# Checkpoints

Populate this directory only with checkpoints produced by actual full-budget runs using `--checkpoint_save_interval 17`.

Required final layout:

```text
checkpoints/
├── accel_repro/
│   ├── 0/
│   ├── 1/
│   └── 2/
└── frontier_accel/
    ├── 0/
    ├── 1/
    └── 2/
```

The student architecture must remain upstream JaxUED's original `ActorCritic`, so organiser evaluation can restore these checkpoints unchanged.
