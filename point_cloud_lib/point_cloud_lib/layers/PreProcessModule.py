import torch

class PreProcessModule(torch.nn.Module):
    """Class to implement a module for pre processing.
    """

    def __init__(self):
        """Constructor.
        """
        self.pre_process_ = False

        # Super class init.
        super(PreProcessModule, self).__init__()

    
    def __process_children_list_start__(self, p_module):
        """Start pre-process.

        Args:
            p_module (Module): module to process children.
        """
        for child_module in p_module.children():
            if isinstance(child_module, PreProcessModule):
                child_module.start_pre_process()
            elif isinstance(child_module, torch.nn.ModuleList):
                self.__process_children_list_start__(child_module)

    
    def __process_children_list_end__(self, p_module):
        """Start pre-process.

        Args:
            p_module (Module): module to process children.
        """
        for child_module in p_module.children():
            if isinstance(child_module, PreProcessModule):
                child_module.end_pre_process()
            elif isinstance(child_module, torch.nn.ModuleList):
                self.__process_children_list_end__(child_module)


    def start_pre_process(self):
        """Start pre-process.
        """
        self.pre_process_ = True
        self.__process_children_list_start__(self)

    
    def end_pre_process(self):
        """Stop pre-process.
        """
        self.pre_process_ = False
        self.__process_children_list_end__(self)