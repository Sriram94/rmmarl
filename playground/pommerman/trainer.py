from crom_implementation import DCROM, DQROM

# Initialize DCROM
dcrom = DCROM(
    n_agents=2,
    state_dim=372,  # Flattened pommerman obs
    action_spaces=[6, 6],
    reward_machines=[PommermanRewardMachine(0), PommermanRewardMachine(1)]
)

# Create trainer with your algorithm
trainer = CROMPommermanTrainer(algorithm='DCROM', n_agents=2)
trainer.agent = dcrom

# Train
trainer.train()
