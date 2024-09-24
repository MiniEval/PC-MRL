import io

from data.cdflib import CDF
import torch

from xml.etree import ElementTree as ET
from data.dataloader import DataLoader
import glob
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

from data.skeleton import Skeleton


class H36MDataLoader(DataLoader):
    def __init__(self, data_folder, skeleton_file, min_length=0, data_device='cpu', skel_device='cpu'):
        self.SCALE = 0.001
        super(H36MDataLoader, self).__init__(data_folder, skeleton_file, min_length, 2, data_device, skel_device)

        # quaternion conversion
        for i in range(len(self.motion_data)):
            data = self.motion_data[i][1::2].clone()
            # rotation_euler = torch.roll(torch.reshape(data[..., 3:], (data.shape[0], -1, 3)), 2, dims=-1)
            rotation_euler = torch.reshape(data[..., 3:], (data.shape[0], -1, 3))
            rotation_matrix = euler_angles_to_matrix(torch.deg2rad(rotation_euler), "ZXY")
            q = torch.reshape(matrix_to_quaternion(rotation_matrix), (data.shape[0], -1))

            sign = torch.gt(torch.linalg.vector_norm(q[1:] - q[:-1], dim=-1, keepdim=True),
                            torch.linalg.vector_norm(q[1:] + q[:-1], dim=-1, keepdim=True)).int()
            sign = (torch.cumsum(sign, dim=0) % 2) * -2 + 1
            q = torch.cat([q[:1], q[1:] * sign], dim=0)
            q = torch.reshape(q, (data.shape[0], -1))

            self.motion_data[i] = torch.cat([data[..., :3] * self.SCALE, q], dim=-1)

    def load_files(self):
        raw_data = []

        for file in glob.glob(self.data_folder + "/**/D3_Angles/*.cdf", recursive=True):
            with open(file, "rb") as f:
                raw_data.append(f.read())

        return raw_data

    def process_data(self, data):
        cdf_data = CDF(io.BytesIO(data))
        motion_data = cdf_data.varget("Pose")[0]

        return torch.tensor(motion_data, dtype=torch.float32)

    @staticmethod
    def parse_array(string, use_float=True):
        elem_list = string.strip("[]").split()
        if use_float:
            elem_list = [float(x) for x in elem_list]
        else:
            elem_list = [int(x) for x in elem_list]

        return elem_list

    @staticmethod
    def setup_skeleton(skeleton_file, device):
        data = ET.parse(skeleton_file)

        root = data.getroot()
        data = root.find("skel_angles").find("tree").findall("item")
        joint_names = []
        joint_hierarchy = []
        joint_offsets = []
        end_sites = {}
        joint_id = 0
        parent_map = dict()
        parent_map[-1] = -1
        for i, joint in enumerate(data):
            name = joint.find("name").text
            parent = parent_map[int(joint.find("parent").text) - 1]
            offset = H36MDataLoader.parse_array(joint.find("offset").text)
            parent_map[i] = joint_id
            if name == "Site":
                end_sites[joint_names[parent]] = offset
            else:
                joint_names.append(name)
                joint_offsets.append(offset)
                joint_hierarchy.append(parent)
                joint_id += 1

        children = [list() for _ in range(len(joint_names))]
        for i, parent_idx in enumerate(joint_hierarchy):
            if parent_idx >= 0:
                children[parent_idx].append(i)

        joint_tails = []

        joint_bgroups = torch.zeros(len(joint_names), 5, dtype=torch.float32, device=device)
        bgroups_spine = ["Hips", "Spine", "Spine1", "Neck", "Head"]
        bgroups_leftarm = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb", "L_Wrist_End"]
        bgroups_rightarm = ["RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumb", "R_Wrist_End"]
        bgroups_leftleg = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
        bgroups_rightleg = ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]

        end_joints = ["LeftHand", "RightHand", "LeftToeBase", "RightToeBase"]

        for i, name in enumerate(joint_names):
            if len(children[i]) == 0:
                joint_tails.append(end_sites[name])
            elif name in ("Hips", "LeftHand", "RightHand"):
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
