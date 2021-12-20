import os
import csv
import rmm
import tempfile
import asyncio

from dask.distributed import Client, Worker, WorkerPlugin

from typing import List


class DependencyInstaller(WorkerPlugin):
    def __init__(self, dependencies: List[str]):
        self._depencendies = " ".join(f"'{dep}'" for dep in dependencies)

    def setup(self, _worker: Worker):
        os.system(f"conda install -c rapidsai-nightly -c rapidsai -c nvidia -c conda-forge -c defaults {self._depencendies}")

# Wrap this in a method used to initialize the module + pass in teh client instance
dependency_installer = DependencyInstaller(["pynvml"])

#client = Client()
#client.register_worker_plugin(dependency_installer)

class RMMResourceMonitor:
    """
    Distributed montor for RMM resource allocations
    """

    def __init__( self, client, outputdir='/tmp'  ):
        self._client = client if isinstance(client, Client) else None
        self._outputdir=outputdir

    def __dispatch__( self, method, **kwargs ):
        if self._client:
            self._client.run( method, **kwargs )
        else:
            return method(*args, **kwargs )

    def get_remote_output_dir( self ):
        return self._outputdir

    def begin_logging( self, prefix="rmmlog"):
        """
        enable rmm logging into dask temporary directory
        """

        def _rmmlogstart( basedir, prefix ):
            import os
            rmm.enable_logging( log_file_name=os.path.join( basedir,  f"{prefix}_" + str(os.getpid())+".log"))

        self.__dispatch__( _rmmlogstart, prefix=prefix, basedir=self.get_remote_output_dir())

    def stop_logging( self ):
        """
        disable rmm logging and mark files for retrieval
        """
        def _rmmlogstop():
            rmm.disable_logging()

        self.__dispatch__( _rmmlogstop )

    def collect( self ):
        """
        distributed command retrieves an logfile
        @return reference to dataframe into which rresults are being loaded
        """
        def _collect():
            for fname in (rmm.get_log_filenames()):
                print( fname )
                #load into memory and return dask_dataframe reference?

        retval = DaskDataframe()
        for lf_future in self.__dispatch__( _collect, localfile ):
            pass

