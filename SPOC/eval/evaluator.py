

class Evaluator:
    """Base class for evaluators."""
    
    def __init__(self, cfg):
        """Initialization"""
        self.cfg = cfg

    def evaluate(self):
        """Evaluate the steps plan for the given task."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _load_agent(self, cfg):
        """Load the planning agent."""

        if cfg.planner.agent_type =="react":
            # load llm handler 
            if 'llm' in cfg:
                from SPOC.llm.agent_llm_handler import ReActLLMHandler
                llm_handler = ReActLLMHandler(cfg)
                # react agent
                from SPOC.planner.react import ReactAgent
                return ReactAgent(cfg, llm_handler)
            elif 'vlm' in cfg:
                NotImplementedError("VLM Planner is not supported yet.")
            else:
                raise ValueError("No valid LLM or VLM configuration found in cfg.")
            
        # Add here for your own safe task planning agent 
        # refer SPOC.planner.react
        else:
            NotImplementedError(f"Agent type {cfg.planner.agent_type} is not implemented.")