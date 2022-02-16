from .eval_hooks import DistEvalHook, EvalHook


# Switch class embeddings of bbox head before start evaluation.
# Bbox head must have 'switch_class' method.

class ZeroShotEvalHook(EvalHook):

    def __init__(self, head_type, *args, **kwargs):
        assert hasattr(head_type, 'switch_class')
        self.head_type = head_type
        super(ZeroShotEvalHook, self).__init__(*args, **kwargs)

    def _do_evaluate(self, runner):
        model = runner.model
        module_list = []
        for name, m in model.named_modules():
            if isinstance(m, self.head_type):
                module_list.append(m)
                m.switch_class('novel')

        super(ZeroShotEvalHook, self)._do_evaluate(runner)
        
        for m in module_list:
            m.switch_class('base')

class ZeroShotDistEvalHook(DistEvalHook):

    def __init__(self, head_type, *args, **kwargs):
        assert hasattr(head_type, 'switch_class')
        self.head_type = head_type
        super(ZeroShotDistEvalHook, self).__init__(*args, **kwargs)

    def _do_evaluate(self, runner):
        model = runner.model
        module_list = []
        for name, m in model.named_modules():
            if isinstance(m, self.head_type):
                module_list.append(m)
                m.switch_class('novel')

        super(ZeroShotDistEvalHook, self)._do_evaluate(runner)
        
        for m in module_list:
            m.switch_class('base')
