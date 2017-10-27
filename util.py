

def change_date_format(src_file='Data/monthly_mean_global_surface_tempreratures_1880-2017.csv',
                       dest_file='Data/monthly_mean_global_surface_tempreratures_1880-2017_new.csv'):
    data = open(src_file, 'r')
    data_new = open(dest_file, 'w')
    counter = -1
    for i, line in enumerate(data):
        if i < 2:
            data_new.write(line)
            continue
        counter += 1
        line_new = line.split(',')
        line_new[0] = line_new[0].split('.')[0] + '-'
        line_new[0] = line_new[0] + '0{}'.format(counter % 12 + 1) if counter % 12 + 1 <= 9 else line_new[
            0] + '{}'.format(counter % 12 + 1)
        data_new.write(','.join(line_new))
    data.close()
    data_new.close()
