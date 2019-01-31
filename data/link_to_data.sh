#!/usr/bin/env bash
# remove old checkpoints if exist
rm data/checkpoints
rm data/val_data

ln -s ~/lyw/data/ghost-network-data/checkpoints data/
ln -s ~/lyw/data/ghost-network-data/val_data data/
