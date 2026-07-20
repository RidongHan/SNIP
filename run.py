import sys
import subprocess


# 超参数调优脚本
def run(command):
    subprocess.call(command, shell=True)

def process_sample():
    for remove_ratio in [0.1,0.2,0.3]:
        for weight in [3,4,5,6,7,8,9,10]:
            for weight_degree in [0,0.1,0.2,0.25,0.5,0.8,0.9,1]:
                for weight_betweenness in [0,0.1,0.2,0.25,0.5,0.8,0.9,1]:
                    for weight_closeness in [0,0.1,0.2,0.25,0.5,0.8,0.9,1]:
                        for weight_eigenvector in [0,0.1,0.2,0.25,0.5,0.8,0.9,1]:
                            if abs(weight_degree + weight_betweenness +
                                    weight_closeness + weight_eigenvector - 1.0) > 1e-6:
                                continue
                                    
                            cmd = 'python main.py ' +' '+ \
                                    ' --remove_ratio ' + str(remove_ratio) + ' '+ \
                                    ' --weight_degree  ' + str(weight_degree ) + ' '+ \
                                    ' --weight_closeness ' + str(weight_closeness) + ' ' + \
                                    ' --weight_eigenvector ' + str(weight_eigenvector) + ' ' + \
                                    ' --weight_betweenness ' + str(weight_betweenness) + ' ' + \
                                    ' --weight ' + str(weight) 
                            run(cmd)
                            sys.stdout.flush()


if __name__ == '__main__':
    process_sample()
