import torch
import clip

def collect_txt(dataset):
    text_map = []
    with open(f'text/{dataset}.txt') as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            temp_list = line.rstrip().lstrip().split(';')
            text_map.append(temp_list)
    text_dict = {}
    if dataset == 'LARA':
        num_text_aug = 5
        for ii in range(num_text_aug):
            if ii == 0:
                text_dict[ii] = [','.join(pasta_list[0:2]) for pasta_list in text_map]
            elif ii == 1:
                text_dict[ii] = [pasta_list[0] + ',' + ','.join(pasta_list[2:3]) for pasta_list in text_map]
            elif ii == 2:
                text_dict[ii] = [pasta_list[0] + ',' + ','.join(pasta_list[3:4]) for pasta_list in text_map]
            elif ii == 3:
                text_dict[ii] = [pasta_list[0] + ',' + ','.join(pasta_list[4:5]) for pasta_list in text_map]
            else:
                text_dict[ii] = [pasta_list[-1] for pasta_list in text_map]
    elif dataset == 'TCG':
        num_text_aug = 5
        for ii in range(num_text_aug):
            if ii == 0:
                text_dict[ii] = [' '.join(pasta_list[0].split('_')) + ',' + ','.join(pasta_list[2:3]) for pasta_list in text_map]
            elif ii == 1:
                text_dict[ii] = [' '.join(pasta_list[0].split('_')) + ',' + ','.join(pasta_list[3:4]) for pasta_list in text_map]
            elif ii == 2:
                text_dict[ii] = [' '.join(pasta_list[0].split('_')) + ',' + ','.join(pasta_list[4:5]) for pasta_list in text_map]
            elif ii == 3:
                text_dict[ii] = [' '.join(pasta_list[0].split('_')) + ',' + ','.join(pasta_list[5:6]) for pasta_list in text_map]
            else:
                text_dict[ii] = [pasta_list[-1] for pasta_list in text_map]
    elif dataset.lower().__contains__('pku'):
        num_text_aug = 5
        for ii in range(num_text_aug):
            if ii == 0:
                text_dict[ii] = [','.join(pasta_list[0:2]) for pasta_list in text_map]
            elif ii == 1:
                text_dict[ii] = [pasta_list[0] + ',' + ','.join(pasta_list[2:4]) for pasta_list in text_map]
            elif ii == 2:
                text_dict[ii] = [pasta_list[0] + ',' + ','.join(pasta_list[4:5]) for pasta_list in text_map]
            elif ii == 3:
                text_dict[ii] = [pasta_list[0] + ',' + ','.join(pasta_list[5:7]) for pasta_list in text_map]
            else:
                text_dict[ii] = [pasta_list[-1] for pasta_list in text_map]
    return text_dict

def collect_txt2(dataset):
    text_map = []
    with open(f'text/{dataset}.txt') as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            temp_list = line.rstrip().lstrip().split(';')
            text_map.append(temp_list)
    text_dict = {}

    if dataset == 'LARA':
        num_text_aug = 5
        for ii in range(num_text_aug):
            if ii == 0:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[-1])) for pasta_list in text_map])
            elif ii == 1:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in text_map])
            elif ii == 2:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[2:3]))) for pasta_list in text_map])
            elif ii == 3:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[3:4]))) for pasta_list in text_map])
            else:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[4:5]))) for pasta_list in text_map])
    elif dataset == 'TCG':
        num_text_aug = 5
        for ii in range(num_text_aug):
            if ii == 0:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[-1])) for pasta_list in text_map])
            # skip head
            elif ii == 1:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[2:3]))) for pasta_list in text_map])
            elif ii == 2:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[3:4]))) for pasta_list in text_map])
            elif ii == 3:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[4:5]))) for pasta_list in text_map])
            else:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[5:6]))) for pasta_list in text_map])
    else:
        num_text_aug = 5
        for ii in range(num_text_aug):
            if ii == 0:
                text_dict[ii] = torch.cat([clip.tokenize((pasta_list[-1])) for pasta_list in text_map])
            elif ii == 1:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in text_map])
            elif ii == 2:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[2:4]))) for pasta_list in text_map])
            elif ii == 3:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[4:5]))) for pasta_list in text_map])
            else:
                text_dict[ii] = torch.cat(
                    [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[5:7]))) for pasta_list in text_map])
    return text_dict