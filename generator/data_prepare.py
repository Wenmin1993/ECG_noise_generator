import os
import wfdb
from IPython.display import display

dl_dir_afdb="C:\\Users\\admin\\Desktop\\论文\\数据库\\afdb"
dl_dir_nsrdb="C:\\Users\\admin\\Desktop\\论文\\数据库\\nsrdb"
dl_dir_nstdb="C:\\Users\\admin\\Desktop\\论文\\数据库\\nstdb"

if __name__ == '__main__':
    wfdb.dl_database('afdb', dl_dir_afdb)
    display(os.listdir(dl_dir_afdb))
    wfdb.dl_database('nsrdb', dl_dir_nsrdb)
    display(os.listdir(dl_dir_nsrdb))
    wfdb.dl_database('nstdb', dl_dir_nstdb)
    display(os.listdir(dl_dir_nstdb))

