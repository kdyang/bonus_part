data1 = open('box_result.txt', 'r')
data2 = open('bonus_ground_truth.txt', 'r')
#data3 = open('meiren/annotations.txt', 'r')
f = open('for_map.txt', 'w')
for line in data1:
    c = 0
    #print('c', c)
    line = line.rstrip()
    words = line.split()
    data3 = open('annotations.txt', 'r')
    for line1 in data3:
        #print(int(words[0]))
        if c == int(words[0]):
            #print('ok')
            line1 = line1.rstrip()
            words1 = line1.split()
            new_line = [words1[0], words[1], words[2], words[3], words[4], words[5], words[6]]
            for line_mem in new_line:
                f.write(str(line_mem) + ' ')
            f.write('\r\n')
            data3.close()
            break
        c += 1
data1.close()
data2.close()
#data3.close()
f.close()