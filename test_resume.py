from main import graph_app
import uuid
thread_id = str(uuid.uuid4())
config={'configurable': {'thread_id': thread_id}}
initial_state = {
    "question_id": "test",
    "question_context": "1+1=",
    "ground_truth": "2",
    "confidence_score": 50,
    "trap_analysis": False
}
print("Updating state as code_executor")
graph_app.update_state(config, initial_state, as_node="code_executor")

print("Testing start...")
for update in graph_app.stream(None, config=config, stream_mode='updates'):
    print(update)

state = graph_app.get_state(config)
print("State Next:", state.next)

# Resume
print("Testing resume...")
graph_app.update_state(config, {"final_decision": "Match"})
print("--- After Update State ---")
for update in graph_app.stream(None, config=config, stream_mode='updates'):
    print(update)

print("Final State Next:", graph_app.get_state(config).next)
