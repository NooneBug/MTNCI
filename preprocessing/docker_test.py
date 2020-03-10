
import docker

dockerfile_path = '../hyperbolics/Docker/'

if __name__ == "__main__":
    client = docker.from_env()
    client.images.build(path = dockerfile_path)
    container = client.containers.run(image = 'hyperbolics/gpu', 
                          name = 'MTNCI-HyperE', 
                          volumes = {'$PWD' : {'bind': '/root/hyperbolics'}},
                          command = 'julia combinatorial/comb.jl -d data/edges/edgelist -m hyper.r64.emb -e 1.0 -p 64 -r 64 -a -s',
                          log_config = {"Type" : "json-file", "Config" : {}},
                          runtime = 'nvidia',
                          detach = True)
    container.logs()
