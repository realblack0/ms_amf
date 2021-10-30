import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--main', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--device', default="cuda")
parser.add_argument('--nohupNNN', required=True)
parser.add_argument('--seed', default="n")
parser.add_argument('--data_dir', default=None)
args = parser.parse_args()

for subject_id in range(1,10):
    CODE = f"nohup python -u {args.main} --name nohup{args.nohupNNN}{subject_id}_{args.name} --subject {subject_id} --device {args.device} --seed {args.seed}"
    CODE += f" --data_dir {args.data_dir}" if args.data_dir else ""
    CODE += f" > nohup{args.nohupNNN}{subject_id}_{args.name}.out"
    
    print(CODE)
    os.system(CODE)
    
print("finish!")

