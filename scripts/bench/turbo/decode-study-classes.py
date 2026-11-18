"""Four fixed greedy decoding classes for acceptance and timing probes."""
PREFIX = 'Follow the user instruction precisely. Produce only the requested answer.'
LANES = [
 'Write a Python function merge_sorted(a, b) that merges two sorted integer lists without calling sorted. Include three assert examples.',
 'Write a vivid story of about 100 words about a lighthouse keeper finding a sealed letter during a storm.',
 'A shop buys 48 boxes with 24 pencils in each. It sells 675 pencils, then receives 17 more boxes of 24. Compute how many pencils remain and explain the arithmetic.',
 'Return only a JSON object describing these two tool calls in order: search_inventory with arguments query="blue notebook" and limit=3; then reserve_item with arguments sku="NB-BLUE-01" and quantity=2. Use the key tool_calls containing objects with name and arguments.'
]
CLASSES=['code','prose','math','tools']
