from fdllm.tooluse import ToolUsePlugin

from .tools import *

class RetrievalPlugin(ToolUsePlugin):
    def __init__(self, datastore, json_contents, json_database):
        super().__init__(
            Tools = [
                QueryCatalogue(datastore=datastore),
                GetContents(json_contents=json_contents),
                GetReferences(json_database=json_database),
                QuerySuppMat(datastore=datastore, json_database=json_database)
            ]
        )
        