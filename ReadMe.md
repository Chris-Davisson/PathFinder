# PathFinder

PathFinder is a Python tool for finding shortest-ish paths in weighted, undirected graphs of any size. It automatically selects the best strategy for your graph, whether it's small enough to fit in memory or large enough to require streaming and sampling.

---

## ✨ Features

* Handles small, large, and huge graphs efficiently
* Uses **Dijkstra's algorithm** for small graphs
* Uses **landmark-based routing** for large graphs
* Enforces time and memory limits to fit requirements
* Progress and error reporting to `stderr`

---

## 📦 Requirements

* Python **3.8+**
* Unix/Linux/Mac (or WSL on Windows)
* Packages: `networkx`, `psutil`, `ijson`

---

## ⚙️ Setup

Clone the repository and enter the folder:

```bash
git clone https://github.com/Chris-Davisson/PathFinder
cd pathfinder
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running PathFinder

```bash
python pathfinder.py -g <graph.json> -q <queries.txt> -o <results.txt>
```

Arguments:

* `-g <graph.json>`: Path to your graph in **node-link JSON** format
* `-q <queries.txt>`: File with queries (each line: `source dest`)
* `-o <results.txt>`: Output file (`-` for stdout)

If `-q` or `-o` are omitted, the program prompts or prints to terminal.

---


## 📄 Example Query File

```
1 42
5 99
100 250
```

---

## 📝 Notes

* For very large graphs, preprocessing may be limited to **60 seconds**.
* On Windows, use **WSL** for full functionality (timers may not work natively though on mine they did).
* Output includes the full path for each query, or a message if no path is found.
* The program path_checker.py is just a helper to make sure the path was real and not a logic error
