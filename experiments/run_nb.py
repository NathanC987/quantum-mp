import nbformat, sys, time
from nbclient import NotebookClient
for f in sys.argv[1:]:
    t=time.time(); nb=nbformat.read(f,4)
    NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':'.'}}).execute()
    nbformat.write(nb,f)
    for c in nb.cells:
        if c.cell_type=='code':
            for o in c.get('outputs',[]):
                if 'text' in o: print(o['text'][:1500])
                elif 'data' in o and 'text/plain' in o['data'] and 'image/png' not in o['data']: print(o['data']['text/plain'][:1500])
    print(f,'time',time.time()-t)
