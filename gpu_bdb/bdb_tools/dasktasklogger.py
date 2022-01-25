import re
import os
import json
import numpy as np

class DaskTaskLogger():
    key_expr=re.compile( '([\w-]+)-([0-9a-f-]{32,36})' )
    
    def __init__(self, client, outputdir='/tmp'):
        self._client=client
        self._outputdir=outputdir

    def mark_begin( self ):
        self._client.get_task_stream()

    def save_tasks( self, prefix='dask' ):
        plotfname=os.path.join(self._outputdir, f"{prefix}_plot.html")
        pdata, pfigure = self._client.get_task_stream(plot='save', filename=plotfname)
        with open( os.path.join(self._outputdir, f"{prefix}_tasks.json"), 'w') as outf:
            json.dump([{k:t[k] for k in filter( lambda x: type(t[x]) != bytes().__class__, t)} for t in pdata],outf)
