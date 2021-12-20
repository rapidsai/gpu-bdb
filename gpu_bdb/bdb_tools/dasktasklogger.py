import re
import os
import json
import numpy as np

example={'worker': 'tcp://10.180.4.206:42643', 'status': 'OK', 'nbytes': 28263552, 'thread': 140534840973056, 'type': b'\x80\x04\x95%\x00\x00\x00\x00\x00\x00\x00\x8c\x13cudf.core.dataframe\x94\x8c\tDataFrame\x94\x93\x94.', 'typename': 'cudf.core.dataframe.DataFrame', 'metadata': {}, 'startstops': ({'action': 'transfer', 'start': 1639787413.9825313, 'stop': 1639787413.998216, 'source': 'tcp://10.180.4.206:45115'}, {'action': 'compute', 'start': 1639787413.998873, 'stop': 1639787414.0106611}), 'key': "('drop-duplicates-combine-d121e7e64a9ef70e5616e411e95f2d3e', 1, 8, 0)"}

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
