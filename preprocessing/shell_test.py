import os

BUILD_PATH = '../hyperbolics/Docker/build.sh'


if __name__ == "__main__":
    os.chdir('../hyperbolics/Docker')
    os.system('docker build -t hyperbolics/gpu .')
    os.chdir('..')
    os.system('nvidia-docker run -v "$PWD:/root/hyperbolics" -it hyperbolics/gpu julia combinatorial/comb.jl -d data/edges/edgelist -m prova.emb -e 1.0 -p 64 -r 64 -a -s')
