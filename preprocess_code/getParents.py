import json
from collections import Counter


def get_data(data_path):
    dict_data = []
    with open(data_path, 'r', encoding='latin-1') as f:
        print("get raw data...")
        raw_data = f.readlines()
        print("Done!")
        for data in raw_data:
            dict_data.append(json.loads(data))
    return dict_data


with open('../JAVA_data/java_NT_id_to_word.txt', 'r') as f:
    id2word = json.loads(f.read())
    word2id = {v: k for k, v in id2word.items()}
    word2id['NONE'] = str(int(word2id['eof']) + 1)


# print(word2id)


# The type of the left sibling
def get_siblings(data):
    siblings = []
    for line in data:
        sibling = ['NONE'] * len(line)
        for node in line:
            if 'children' in node and len(node['children']) > 1:
                for index in range(1, len(node['children'])):
                    sibling[node['children'][index]] = line[node['children'][index - 1]]['type']
        siblings.append(sibling)
    return siblings


def get_sibling_list(data_path, word2id):
    siblings = []
    with open(data_path, 'r', encoding='latin-1') as f:
        print("get raw data...")
        raw_data = f.readlines()
    for line in raw_data:
        line = json.loads(line)
        sibling = [[int(word2id['NONE'])]] * len(line)
        for node in line:
            if 'children' in node and len(node['children']) > 1:
                for index in range(1, len(node['children'])):
                    sibling[node['children'][index]] = [int(word2id[line[node['children'][i]]['type']]) for i in
                                                        range(index)]
        siblings.append(sibling)
    return siblings


# train_sibling_list = get_sibling_list('../py/python100k_train.json',word2id)
# print(train_sibling_list[0])
# with open('../py/py_train_sibling_list.json', 'w') as f:
#     f.write(json.dumps(train_sibling_list))

# print("Getting siblings...")
# valid_siblings = get_siblings(valid_data)
# train_siblings = get_siblings(train_data)
# print("Done")

def write_siblings(filename, siblings, word2id):
    sibling_ids = []
    for line in siblings:
        sibling_ids.append([int(word2id[item]) for item in line])
    with open(filename, 'w') as f:
        f.write(json.dumps(sibling_ids))


def get_syntax_list(sibling_file, par_file, target_file):
    sibling_data = json.loads(open(sibling_file, 'r').read())
    par_data = json.loads(open(par_file, 'r').read())
    syntax_list = []
    for i in range(len(sibling_data)):
        syntax_list_line = []
        for j in range(len(sibling_data[i])):
            data = [int(par_data[i][j][0])]
            data.extend(sibling_data[i][j])
            syntax_list_line.append(data)
        syntax_list.append(syntax_list_line)
    print(syntax_list[0])
    with open(target_file, 'w') as f:
        f.write(json.dumps(syntax_list))


# get_syntax_list('../py/py_train_sibling_list.json','../py/train_par_path.txt', '../py/train_syntax_list.txt')

def get_avg_len(syntax_list_file):  # PY: 15
    syntax_list = json.loads(open(syntax_list_file, 'r').read())
    total = 0
    sum = 0
    for line in syntax_list:
        for item in line:
            total += 1
            sum += len(item)
    print(sum / total)


# print("Writing siblings...")
# write_siblings('../JAVA_data/valid_sibling.txt', valid_siblings, word2id)
# print("Done")


# with open('../JAVA_data/valid_sibling.txt', 'r') as f:
#     train_siblings = json.loads(f.read())
#     total = 0
#     non_sib = 0
#     for i in range(len(train_siblings)):
#         for j in range(len(train_siblings[i])):
#             total += 1
#             if train_siblings[i][j] == 176:
#                 non_sib += 1
# print(non_sib/total)


def get_children(node, line, layer, ans):
    if ans[layer] == []:
        ans[layer] = []
    ans[layer].append(node['type'])
    if 'children' in node:
        nodes = [line[i] for i in node['children']]
        for n in nodes:
            get_children(n, line, layer + 1, ans)
    else:
        return


def get_children_layer(node, line, layer):
    node['layer'] = layer
    if 'children' in node:
        nodes = [line[i] for i in node['children']]
        for n in nodes:
            get_children_layer(n, line, layer + 1)
    else:
        return


def get_layer(data, fname):
    layers = []
    for code in data:
        get_children_layer(code[0], code, 0)
        layer = []
        for node in code:
            layer.append(node['layer'])
        layers.append(layer)
    with open(fname, 'w') as f:
        f.write(json.dumps(layers))
    return layers


# with open('js_train_par_path1.txt','r') as f:
#     train1 = json.loads(f.read())
# with open('js_train_par_path2.txt','r') as f:
#     train2 = json.loads(f.read())
# train1.extend(train2)
# with open('js_train_par_path.txt','w') as f:
#     f.write(json.dumps(train1))

# valid_layers = get_layer(valid_data, '../py/valid_layer.txt')
# train_layers = get_layer(train_data, '../py/train_layer.txt')
# print(layers)

# import numpy as np
# with open('../py/train_layer.txt', 'r') as f:
#     train_layers = json.loads(f.read())
#     print(max(np.concatenate(train_layers,0)))
#
# with open('../py/valid_layer.txt', 'r') as f:
#     valid_layers = json.loads(f.read())
#     print(max(np.concatenate(valid_layers,0)))


def get_layers_elements(data):
    all_layers_data = []
    for code in data:
        layers = [[]] * 500
        get_children(code[0], code, 0, layers)
        Index = layers.index([])
        layers = layers[:Index]
        all_layers_data.append(layers)
    return all_layers_data


def add_par_att(code):
    for i in range(len(code)):
        try:
            if 'children' in code[i]:
                for index in code[i]['children']:
                    code[index]['parent_index'] = i
        except:
            pass
    return code


def get_parent_node(code, node, par):
    try:
        if 'parent_index' in node:
            # print(node['parent_index'])
            par.append(int(word2id[code[node['parent_index']]['type']]))
            # print(par)
            get_parent_node(code, code[node['parent_index']], par)
    except:
        pass
    return par


def pad_path(path, path_len, pad_id):
    if len(path) > path_len:
        path = path[:path_len]
    else:
        path += [pad_id] * (path_len - len(path))
    return path


def get_pad_syntax_list(syntax_file, pad_len, pad_id, target_file):
    syntax_list = json.loads(open(syntax_file, 'r').read())
    padded_syntax_list = []
    for line in syntax_list:
        line_syntax = []
        for item in line:
            padded_item = pad_path(item, pad_len, pad_id)
            line_syntax.append(padded_item)
        padded_syntax_list.append(line_syntax)
    with open(target_file, 'w') as f:
        f.write(json.dumps(padded_syntax_list))


def get_par_data(data_path, path_len=None, pad_id=None):
    with open('../py/py_valid_par_path_3.txt', 'w') as wf:
        with open(data_path, 'r', encoding='latin-1') as f:
            print("get raw data...")
            raw_data = f.readlines()
            print("Done!")
        for code in raw_data:
            par_code = add_par_att(json.loads(code))  # add par_index attrubutes
            # print(par_code)
            code_par = []
            for node in par_code:
                node_par = []
                node_par = get_parent_node(par_code, node, node_par)
                node_par = pad_path(node_par, path_len, pad_id)
                code_par.append(node_par)
            wf.write(json.dumps(code_par) + '\n')


get_par_data('../py/python50k_eval.json', 5, word2id['eof'])

# -------------------------compute path avg length--------------
# par_path_len = []
# def get_par_data(data_path, path_len=None, pad_id=None):
#     with open(data_path, 'r', encoding='latin-1') as f:
#         print("get raw data...")
#         raw_data = f.readlines()
#         print("Done!")
#     parents = []
#     for code in raw_data:
#         par_code = add_par_att(json.loads(code))  # add par_index attrubutes
#         # print(par_code)
#         code_par = []
#         for node in par_code:
#             node_par = []
#             node_par = get_parent_node(par_code, node, node_par)
#             # node_par = pad_path(node_par, path_len, pad_id)
#             code_par.append(node_par)
#
#         for node in code_par:
#             # print(node)
#             par_path_len.append(len(node))
#
#         # parents.append(code_par)
#     return parents
#
# par_path_len = []
# parents = get_par_data('../JAVA_data/json_data/train.json')
#
# for code in parents:
#     for node in code:
#         # print(node)
#         par_path_len.append(len(node))
# import numpy as np
# with open('java_train_path.txt','w') as f:
#     f.write(json.dumps(par_path_len))
# print(np.average(par_path_len))  # PY 7 JS 15 JAVA 5

# import matplotlib.pyplot as plt
# with open('py_train_path.txt','r') as f:
#     par_path_len = json.loads(f.read())
#     # f.write(json.dumps(par_path_len))
# fig = plt.figure()
# ax = plt.subplot()
# ax.boxplot([par_path_len], whis=[5, 95])
# plt.show()


# ---------------------------------------------------

# parents = get_par_data(train_data, 5, word2id['eof'])
# print(parents[0])
# with open('../js/train_par_path.txt','w') as f:
#     f.write(json.dumps(parents))


# code_par = []
# for node in par_data[0]:
#     node_par = []
#     node_par = get_parent_node(par_data[0], node, node_par)
#     code_par.append(node_par)
# print(code_par)


# def get_grammers(file_name):
#     grammers = []
#     with open(file_name, 'r', encoding='latin-1') as f:
#         print("get raw data...")
#         raw_data = f.readlines()
#     for line in raw_data:
#         code = json.loads(line)
#         for node in code:
#             if node['type'] == 'Module' or node['type'] == 'body':
#                 continue
#             grammer = []
#             grammer.append(node['type'])
#             if 'children' in node:
#                 childrens = node['children']
#                 grammer.append(code[childrens[0]]['type'])
#                 # for index in childrens:
#                 #     if code[index]['type'] == grammer[-1]:
#                 #         continue
#                 #     grammer.append(code[index]['type'])
#             if grammer not in grammers:
#                 # print(grammer)
#                 grammers.append(grammer)
#     return grammers
#
# grammers = get_grammers('../py/python100k_train.json') # total 230782
# print(grammers[:1000])
# print(len(grammers))
