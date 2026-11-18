"""Independent prose/code/JSON/edit requests; two sequential passes are cold/warm."""
PREFIX = 'Follow the user instruction precisely. Produce only the requested answer.'
LANES = [
    'Write a Python function merge_sorted(a, b) that merges two sorted integer lists without calling sorted. Include three assert examples.',
    'Write a vivid story of about 100 words about a lighthouse keeper finding a sealed letter during a storm.',
    'Return only a JSON object describing these two tool calls in order: search_inventory with arguments query="blue notebook" and limit=3; then reserve_item with arguments sku="NB-BLUE-01" and quantity=2. Use the key tool_calls containing objects with name and arguments.',
    'Edit this Python function to handle an empty list and preserve the original input. Return only the corrected function:\n\ndef median(xs):\n    xs.sort()\n    return xs[len(xs)//2]\n\nFor an even number of values return the mean of the two central values. Raise ValueError for an empty list.',
]
CLASSES = ['code', 'prose', 'json', 'edit']
