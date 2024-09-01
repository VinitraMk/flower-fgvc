

def cleanup_strings(a):
    #a = a.replace("'", "")
    a = a.replace("\n", "")
    return a

def cleanup_descriptions(a):
    a = a.replace("\n", "")
    a = a.lower()
    return a

def get_descriptions(descs, classnames, class_desc):
    
    for i, lbl in enumerate(classnames):
        found_description = False
        for desc in descs:
            if lbl in desc:
                d = desc.split(':')[1]
                class_desc[i] = {
                    'label': lbl,
                    'description': d[1:]
                }
                found_description = True
                break
        if not(found_description) and class_desc[i]['description'] == '':
            class_desc[i] = {
                'label': lbl,
                'description': ''
            }
    return class_desc
        


classnames = []

with open('./data/classnames.txt') as fp:
    classnames = fp.readlines()

classnames = list(map(cleanup_strings, classnames)) 
print('classes', classnames)

descs1, descs2 = [], []
with open('./data/class_descriptions.txt') as fp:
    descs1 = fp.readlines()

with open('./data/class_descriptions_1.txt') as fp:
    descs2 = fp.readlines()
    
descs1 = list(map(cleanup_descriptions, descs1))
descs2 = list(map(cleanup_descriptions, descs2))

class_desc = [{'label': '', 'description': ''}] * 102

class_desc = get_descriptions(descs1, classnames, class_desc)
class_desc = get_descriptions(descs2, classnames, class_desc)

missing_descs = [el['label'] for el in class_desc if el['description'] == '']

print('no of missing descriptions', len(missing_descs))
print(missing_descs)

with open('./data/processed_class_descriptions.txt', 'w+') as fp:
    for el in class_desc:
        line = f'{el["label"]} : {el["description"]}\n'
        fp.write(line)