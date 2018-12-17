from fabric import task
from fabric import Connection

PROJECT_ROOT = {
    'ipf_cluster': '~/projects/tdse/',
    'cluster': '~/share/tdse/'
}

@task
def update_code(conn):
    root = PROJECT_ROOT[conn.original_host]

    with conn.cd(root):
        res = conn.run("git stash")
        conn.run("git pull origin master", pty=True)
        if res.stdout != "No local changes to save\n":
            conn.run("git stash pop")
        conn.run("make")

@task
def rebuild_cython(conn):
    root = PROJECT_ROOT[conn.original_host]

    with conn.cd(root):
        conn.run("rm wrapper/*.cpp")
        conn.run("make")
