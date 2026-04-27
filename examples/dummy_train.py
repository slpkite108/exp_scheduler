import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--sleep", type=float, default=0.5)
args = parser.parse_args()

print(f"start lr={args.lr} batch_size={args.batch_size} epochs={args.epochs}", flush=True)
for epoch in range(1, args.epochs + 1):
    time.sleep(args.sleep)
    print(f"epoch {epoch}/{args.epochs}", flush=True)
print("done", flush=True)
