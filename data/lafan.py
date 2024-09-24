import torch

from data.bvh import parse_bvh_skeleton
from data.dataloader import DataLoader
import glob
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

from data.skeleton import Skeleton


class LaFANDataLoader(DataLoader):
    def __init__(self, data_folder, skeleton_file, min_length=0, data_device='cpu', skel_device='cpu'):
        self.SCALE = 0.01
        super(LaFANDataLoader, self).__init__(data_folder, skeleton_file, min_length, 2, data_device, skel_device)

        # quaternion conversion
        for i in range(len(self.motion_data)):
            data = self.motion_data[i][1::2].clone()
            # rotation_euler = torch.flip(torch.reshape(data[..., 3:], (data.shape[0], -1, 3)), [2])
            rotation_euler = torch.reshape(data[..., 3:], (data.shape[0], -1, 3))
            rotation_matrix = euler_angles_to_matrix(torch.deg2rad(rotation_euler), "ZYX")
            q = matrix_to_quaternion(rotation_matrix)

            _xzy_to_xyz = torch.sqrt(torch.tensor([[2, 2, 0, 0]], dtype=torch.float32, device=data_device)) / 2
            _xzy_to_xyz = _xzy_to_xyz.expand(q[:, 0].shape)
            q[:, 0] = self.skeleton._qmul(_xzy_to_xyz, q[:, 0])
            data[..., :3] = self.skeleton._qrot(data[..., :3], _xzy_to_xyz)

            sign = torch.gt(torch.linalg.vector_norm(q[1:] - q[:-1], dim=-1, keepdim=True),
                            torch.linalg.vector_norm(q[1:] + q[:-1], dim=-1, keepdim=True)).int()
            sign = (torch.cumsum(sign, dim=0) % 2) * -2 + 1
            q = torch.cat([q[:1], q[1:] * sign], dim=0)
            q = torch.reshape(q, (data.shape[0], -1))

            self.motion_data[i] = torch.cat([data[..., :3] * self.SCALE, q], dim=-1)

    def load_files(self):
        raw_data = []

        for file in glob.glob(self.data_folder + "/**/*.bvh", recursive=True):
            with open(file, "r") as f:
                while f.readline() != "MOTION\n":
                    continue
                raw_data.append(f.read())

        return raw_data

    def process_data(self, data):
        lines = data.split('\n')
        motion_data = []
        for line in lines[2:]:
            words = line.strip().split(" ")
            try:
                motion_data.append(list(map(float, words)))
            except ValueError:
                pass

        return torch.tensor(motion_data, dtype=torch.float32)

    @staticmethod
    def setup_skeleton(skeleton_file, device):
        with open(skeleton_file, "r") as f:
            joint_names, joint_offsets, joint_hierarchy, end_sites = parse_bvh_skeleton(f.read())

        children = [list() for _ in range(len(joint_names))]
        for i, parent_idx in enumerate(joint_hierarchy):
            if parent_idx >= 0:
                children[parent_idx].append(i)

        joint_tails = []

        joint_bgroups = torch.zeros(len(joint_names), 5, dtype=torch.float32, device=device)
        bgroups_spine = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"]
        bgroups_leftarm = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
        bgroups_rightarm = ["RightShoulder", "RightArm", "RightForeArm", "RightHand"]
        bgroups_leftleg = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe"]
        bgroups_rightleg = ["RightUpLeg", "RightLeg", "RightFoot", "RightToe"]

        end_joints = ["LeftHand", "RightHand", "LeftToe", "RightToe"]

        for i, name in enumerate(joint_names):
            if len(children[i]) == 0:
                joint_tails.append(end_sites[name])
            elif name == "Hips":
                joint_tails.append(joint_offsets[children[i][-1]])
            else:
                joint_tails.append(joint_offsets[children[i][0]])

            if name in bgroups_spine:
                joint_bgroups[i, 0] = 1
            elif name in bgroups_leftarm:
                joint_bgroups[i, 1] = 1
            elif name in bgroups_rightarm:
                joint_bgroups[i, 2] = 1
            elif name in bgroups_leftleg:
                joint_bgroups[i, 3] = 1
            elif name in bgroups_rightleg:
                joint_bgroups[i, 4] = 1

        joint_offsets = torch.tensor(joint_offsets, dtype=torch.float32, device=device)
        joint_tails = torch.tensor(joint_tails, dtype=torch.float32, device=device)

        pairs = []
        for i in range(len(bgroups_leftarm)):
            pairs.append((joint_names.index(bgroups_leftarm[i]), joint_names.index(bgroups_rightarm[i])))

        for i in range(len(bgroups_leftleg)):
            pairs.append((joint_names.index(bgroups_leftleg[i]), joint_names.index(bgroups_rightleg[i])))

        return Skeleton(joint_names, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, end_joints, pairs,
                        scale=0.01, device=device)
