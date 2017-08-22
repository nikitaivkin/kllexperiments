import paramiko
import os

def readHosts(hostfile):
    return tuple(open(hostfile, 'r'))

def sshConnect(hostname):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname,username='ubuntu',
                key_filename='''/home/local/ANT/ivkin/pssh/kllTesting.pem''')
    return ssh

def sshParConnect(hostfile):
    sshList = []
    for hostname in readHosts(hostfile):
        sshList.append(sshConnect(hostname.rstrip()))
    return sshList

def sshParClose(sshList):
    for ssh in sshList:
        ssh.close()

def sshRequest(ssh, request):
    stdin, stdout, stderr = ssh.exec_command(request)
    stdin.flush()
    # out = stdout.read().splitlines() + stderr.read().splitlines()
    return (stdout, stderr)

def sshParRequest(sshList, request):
    outList = []
    for ssh in sshList:
        outList.append(sshRequest(ssh, request))
    return outList

def printOutList(outList):
    for std_i, (stdout, stderr) in enumerate(outList):
        print "#" * 20 + "\nNode #" + str(std_i) + "\n" + "#"*20
        for line in stdout.read().splitlines() + stderr.read().splitlines():
            print line

def printOutList2File(outList, filedD):
    for std_i, (stdout, stderr) in enumerate(outList):
        for line in stdout.read().splitlines() + stderr.read().splitlines():
            fileD.write(line + "\n")
            fileD.flush()


def sshWaitToFinish(outList):
    for std_i, (stdout, stderr) in enumerate(outList):
        res = stdout.read().splitlines() + stderr.read().splitlines()



def sshSendFile(ssh, localpath, remotepath):
    sftp = ssh.open_sftp()
    sftp.put(localpath, remotepath)
    sftp.close()


def sshGetFile(ssh, localpath, remotepath):
    sftp = ssh.open_sftp()
    sftp.get(remotepath, localpath)
    sftp.close()

def sshSendFolder(ssh, localpath, remotepath):
    sftp = ssh.open_sftp()
    files = os.listdir(localpath)
    for file in files:
        sshSendFile(ssh, os.path.join(localpath, file), os.path.join(remotepath, file))


def sshGetFolder(ssh, localpath, remotepath):
    sftp = ssh.open_sftp()
    files = sftp.listdir(remotepath)
    for file in files:
        sshGetFile(ssh, os.path.join(localpath, file), os.path.join(remotepath, file))


def sshParSendFile(sshList, localpath, remotepath):
    for ssh in sshList:
        sshSendFile(ssh, localpath, remotepath)


def sshParGetFile(sshList, localpath, remotepath):
    for ssh in sshList:
        sshGetFile(ssh, localpath, remotepath)

def sshParSendFolder(sshList, localpath, remotepath):
    for ssh in sshList:
        sshSendFolder(ssh, localpath, remotepath)

def sshParGetFolder(sshList, localpath, remotepath):
    for ssh in sshList:
        sshGetFolder(ssh, localpath, remotepath)




if __name__ == "__main__":
    hostfile = '''/home/local/ANT/ivkin/pssh/nodes'''
    sshList = sshParConnect(hostfile)
    # sshParSendFolder(sshList, "/home/local/ANT/ivkin/pssh", "/home/ubuntu/folder1/folder2")
    # sshParGetFolder(sshList, "/home/local/ANT/ivkin/pssh/new/", "/home/ubuntu/folder1/folder2")

    # outList = sshParRequest(sshList, "sudo add-apt-repository ppa:fkrull/deadsnakes; \n")
    # printOutList(outList)
    # outList = sshParRequest(sshList,"\n sudo apt-get update; \n")
    # printOutList(outList)
    outList = sshParRequest(sshList,"export DEBIAN_FRONTEND=noninteractive;  sudo apt-get -y install python-minimal; ")
    printOutList(outList)
    # outList = sshParRequest(sshList, "Y")
    # printOutList(outList)

    # outList = sshParRequest(sshList, "tmux a -t kll")
    # outList = sshParRequest(sshList, "tmux detach")
    # outList = sshParRequest(sshList, "sleep 2")
    # outList = sshParRequest(sshList, "sleep 7; mkdir folder1")
    # sshWaitToFinish(outList)
    # outList = sshParRequest(sshList, "mkdir folder1/folder2")
    printOutList(outList)
    sshParClose(sshList)

