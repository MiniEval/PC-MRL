def parse_bvh_skeleton(data):
    joint_names = []
    joint_offsets = []
    end_sites = {}
    joint_hierarchy = [-1]

    lines = data.split("\n")
    l = 0
    stack = []
    joint_id = 0
    while not lines[l].startswith("ROOT"):
        l += 1

    joint_names.append(lines[l].split(" ")[1])
    l += 1
    while l < len(lines):
        words = lines[l].lstrip().split(" ")

        if words[0] == "MOTION":
            break

        if words[0] == "OFFSET":
            joint_offsets.append(list(map(float, words[1:])))
        if words[0] == "JOINT":
            joint_names.append(words[1])
            joint_hierarchy.append(stack[-1])
            joint_id += 1
        if words[0] == "{":
            stack.append(joint_id)
        if words[0] == "}":
            stack.pop()
        if words[0] == "End":
            end_sites[joint_names[-1]] = list(map(float, lines[l+2].split(" ")[1:]))
            l += 3
        l += 1

    return joint_names, joint_offsets, joint_hierarchy, end_sites