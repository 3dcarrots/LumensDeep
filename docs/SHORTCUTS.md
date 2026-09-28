# Lumens Deep Keyboard & Mouse Shortcuts Cheat Sheet

Quick reference for all navigation, editing, AI, and tool shortcuts in **Lumens Deep**.

---

## 1. File & Project Shortcuts

| Shortcut | Command | Description |
| :--- | :--- | :--- |
| `Ctrl + O` | Open File | Open a RAW or standard image file |
| `Ctrl + S` | Save Adjustments | Save non-destructive JSON sidecar settings |
| `Ctrl + E` | Export Image | Render and export processed image |
| `Ctrl + Q` | Quit | Close the application |

---

## 2. Navigation & Canvas View

| Shortcut / Action | Command | Description |
| :--- | :--- | :--- |
| `Mouse Wheel` | Zoom In / Out | Zoom smoothly centered at cursor position |
| `Ctrl + +` / `Ctrl + =` | Zoom In | Increase zoom level |
| `Ctrl + -` | Zoom Out | Decrease zoom level |
| `Ctrl + 0` | Fit View | Fit the entire photo into the viewport |
| `Ctrl + 1` | 100% (1:1 View) | Zoom to 1:1 pixel scale |
| `Left Click + Drag` | Pan | Pan canvas (when brush tool is inactive) |
| `Middle Click + Drag`| Pan | Pan canvas in any mode |

---

## 3. Tool & Editing Shortcuts

| Shortcut | Tool | Description |
| :--- | :--- | :--- |
| `C` | Crop & Straighten | Toggle interactive Rule-of-Thirds crop HUD |
| `B` | Brush Tool | Toggle GPU layer mask painting tool |
| `[` / `]` | Brush Size | Decrease / increase brush radius |
| `Right Click + Drag` | Erase Brush | Erase painted mask under cursor |
| `\` | Before / After | Toggle view of original unprocessed RAW image |
| `Ctrl + Z` | Undo | Undo last action |
| `Ctrl + Y` / `Ctrl + Shift + Z` | Redo | Redo last undone action |

---

## 4. AI Assistant & 3D Neural Map Shortcuts

| Shortcut | Command | Description |
| :--- | :--- | :--- |
| `Ctrl + Shift + G` | AI Auto-Correction | Run Gemini iterative auto-correction pass on active photo |
| `Ctrl + Shift + A` | Batch AI Auto-Correction | Run Batch AI Auto-Correction sequentially on Filmstrip |
| `Ctrl + Shift + F` | Full Auto Workflow | 1-Click Culling + 3D Assessment + Batch AI Processing |
| `Escape` | Stop AI Processing | Cancel active Gemini AI pass or batch workflow immediately |
| `Enter` *(in AI Prompt)* | Send AI Prompt | Execute one-time directive and auto-clear input field |
| `Shift + Enter` *(in AI Prompt)* | Newline in Prompt | Insert a line break in prompt input box |
| `Ctrl + Shift + C` | AI Batch Culling | Launch 2-Phase AI Photo Culling & Quality Rating dialog |
| `Ctrl + Shift + D` | Compute Depth Map | Estimate neural 16-bit metric depth map via Marigold |
| `Ctrl + Shift + N` | Compute Normal Map | Estimate high-frequency surface normal vectors |
| `Ctrl + Shift + S` | Compute Subject Mask | Generate neural subject alpha matting mask |
| `D` | View Depth Map | Toggle 3D depth map visualizer |
| `N` | View Normal Map | Toggle surface normals visualizer |
| `S` | View Subject Mask | Toggle subject foreground visualizer |
| `M` | View Layer Mask | Toggle active layer mask overlay |
| `V` | Fast Viewer Mode | Toggle instant sub-50ms preview browse mode |

---

## 5. Filmstrip, Rating & Culling Shortcuts

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `1` – `5` | Rate Photo | Assign 1 to 5 Star rating to active photo |
| `0` | Clear Rating | Reset photo rating to 0 (unrated) |
| `X` | Toggle Reject | Mark or unmark photo for deletion (`❌`) |
| `Left Arrow` / `Right Arrow` | Previous / Next | Navigate to adjacent photo in Filmstrip |
| `Home` / `End` | First / Last | Jump to first or last photo in current folder |
