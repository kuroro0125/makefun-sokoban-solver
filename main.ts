// Push-optimal Sokoban solver (original implementation of standard algorithms).
//
// Why push-optimal: scoring pays 1000 points only at the optimal push count P*
// and under 500 for even one push over, so the search must prove push-optimality.
//
// Design:
//  * Push-level A*: a state is the sorted box squares plus the player's
//    reachable region, named by its lowest square (BFS/DFS flood between
//    pushes). The player's walk between pushes is a shortest BFS path.
//  * Admissible heuristic: minimum-cost box->goal matching, where pairing a
//    box with a goal costs the exact pushes that box alone needs to reach it
//    (reverse pull BFS per goal, other boxes ignored). Exact subset DP for
//    up to 8 boxes/goals; above that, the max of the per-box and per-goal
//    minimum sums (still admissible).
//  * Sound deadlock pruning: dead squares (no goal reachable in the pull
//    graph), frozen 2x2 blocks, and a freeze fixpoint (a box is immobile
//    when each axis has a side blocked by a wall or an immobile box; an
//    immobile box off its goal is lost).
//  * Room decomposition: cut the player's row (the hallway) out of the
//    floor; each remaining component with boxes is a room. Boxes can never
//    usefully enter the hallway (its squares are dead for boxes), so rooms
//    are independent: solve each room push-optimally on its own sub-board
//    (room squares + hallway squares walkable but forbidden to boxes) and
//    sum. Levels that do not decompose fall back to whole-level A*.
//  * Determinism: budgets are counted in created states, never in time;
//    all ties break by fixed orders (square order, directions u,d,l,r,
//    insertion order). No clock or randomness is read.
//
// I/O: read MAKEFUN_INPUT_DIR (default /input), one directory per instance
// holding level.txt (XSB); write MAKEFUN_OUTPUT_DIR (default /output)
// /<id>/solution.txt with the moves (u/d/l/r, uppercase = push).

import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order u, d, l, r; index k^1 is the opposite direction.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const WALK = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];
const INF = 1e9;
// State budgets (created nodes), per room and per whole-level search.
const ROOM_STATES = 800000;
const LEVEL_STATES = 800000;

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, size, wall, goals, boxes, player };
}

// A searchable board: neighbour table, goal set, per-goal push distances
// (reverse pull BFS), and the live mask (some goal reachable).
function makeBoard(w, h, wall, goals) {
  const size = w * h;
  const nbr = new Int32Array(size * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let k = 0; k < 4; k++) {
        const nx = x + DX[k];
        const ny = y + DY[k];
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) nbr[i * 4 + k] = ny * w + nx;
      }
    }
  }
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dist = new Int32Array(size * Math.max(goals.length, 1)).fill(-1);
  const live = new Uint8Array(size);
  const queue = new Int32Array(size);
  goals.forEach((g, gi) => {
    const base = gi * size;
    let qh = 0;
    let qt = 0;
    queue[qt++] = g;
    dist[base + g] = 0;
    while (qh < qt) {
      const c = queue[qh++];
      for (let k = 0; k < 4; k++) {
        // A box at c was pushed there from y = c - dir, player at y - dir.
        const y = nbr[c * 4 + (k ^ 1)];
        if (y < 0 || wall[y] || dist[base + y] >= 0) continue;
        const stand = nbr[y * 4 + (k ^ 1)];
        if (stand < 0 || wall[stand]) continue;
        dist[base + y] = dist[base + c] + 1;
        queue[qt++] = y;
      }
    }
    for (let c = 0; c < size; c++) if (dist[base + c] >= 0) live[c] = 1;
  });
  return { w, h, size, wall, goals, goalSet, nbr, dist, live };
}

// Flood the player's reachable region from `from` around walls and occupied
// squares, stamping cells with `stamp`; returns the lowest reachable square.
function flood(board, occupied, from, seen, stamp, stack) {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = board.nbr[p * 4 + k];
      if (q >= 0 && !board.wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Shortest walk from `from` to `to` around walls/boxes; null if unreachable.
function walkPath(board, occupied, from, to) {
  if (from === to) return '';
  const size = board.size;
  const prev = new Int32Array(size).fill(-1);
  const how = new Int8Array(size).fill(-1);
  const queue = new Int32Array(size);
  let qh = 0;
  let qt = 0;
  queue[qt++] = from;
  prev[from] = from;
  while (qh < qt) {
    const p = queue[qh++];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = board.nbr[p * 4 + k];
      if (q < 0 || board.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue[qt++] = q;
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = WALK[how[p]] + s;
  return s;
}

// Deadlock test for a box configuration; `moved` is the box just pushed.
function isDeadlocked(board, occupied, boxes, moved) {
  const { wall, goalSet, nbr, w, size } = board;
  // Frozen 2x2 blocks containing the moved box.
  for (const a of [moved, moved - 1, moved - w, moved - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= size) continue;
    // Guard against row wrap for the a-1 / a-w-1 origins.
    if ((a % w) === w - 1) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
  }
  // Freeze fixpoint.
  const immobile = new Uint8Array(size);
  const blocked = (c) => c < 0 || wall[c] === 1 || (occupied[c] === 1 && immobile[c] === 1);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (immobile[b]) continue;
      const xDead = blocked(nbr[b * 4 + 2]) || blocked(nbr[b * 4 + 3]);
      const yDead = blocked(nbr[b * 4 + 0]) || blocked(nbr[b * 4 + 1]);
      if (xDead && yDead) {
        immobile[b] = 1;
        changed = true;
      }
    }
  }
  for (const b of boxes) if (immobile[b] && !goalSet[b]) return true;
  return false;
}

// Admissible heuristic: exact min-cost matching (subset DP) up to 8 boxes,
// else max of the per-box / per-goal minimum sums. INF = no finite matching.
function heuristic(board, boxes) {
  const { dist, goals, size } = board;
  const n = boxes.length;
  if (n === 0) return 0;
  if (n <= 8 && goals.length <= 8) {
    if (n > goals.length) return INF;
    const full = (1 << goals.length) - 1;
    const dp = new Float64Array(1 << goals.length).fill(INF);
    dp[0] = 0;
    let answer = INF;
    for (let mask = 0; mask <= full; mask++) {
      const cur = dp[mask];
      if (cur >= INF) continue;
      let k = 0;
      for (let m = mask; m; m &= m - 1) k++;
      if (k === n) {
        if (cur < answer) answer = cur;
        continue;
      }
      if (k > n) continue;
      const b = boxes[k];
      for (let j = 0; j < goals.length; j++) {
        if (mask & (1 << j)) continue;
        const c = dist[j * size + b];
        if (c < 0) continue;
        const nm = mask | (1 << j);
        if (cur + c < dp[nm]) dp[nm] = cur + c;
      }
    }
    return answer;
  }
  let byBox = 0;
  for (const b of boxes) {
    let best = INF;
    for (let j = 0; j < goals.length; j++) {
      const c = dist[j * size + b];
      if (c >= 0 && c < best) best = c;
    }
    if (best >= INF) return INF;
    byBox += best;
  }
  let byGoal = 0;
  for (let j = 0; j < goals.length; j++) {
    let best = INF;
    for (const b of boxes) {
      const c = dist[j * size + b];
      if (c >= 0 && c < best) best = c;
    }
    if (best >= INF) return INF;
    byGoal += best;
  }
  return Math.max(byBox, byGoal);
}

// Push-optimal A* on one board. `forbidden` squares may be walked on but a
// box may never enter them. Returns the move string, or null on budget
// exhaustion / unsolvability.
function solveBoard(board, startBoxes, startPlayer, forbidden, maxStates) {
  const { wall, goalSet, live, nbr } = board;
  const size = board.size;
  if (startBoxes.every((b) => goalSet[b])) return '';
  if (startBoxes.some((b) => !live[b] || forbidden[b])) return null;
  const h0 = heuristic(board, startBoxes);
  if (h0 >= INF) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;
  const keyOf = (boxes, low) => String.fromCharCode(low, ...boxes);

  for (const b of startBoxes) occupied[b] = 1;
  const startLow = flood(board, occupied, startPlayer, seen, ++stamp, stack);
  occupied.fill(0);

  const nodeBoxes = [startBoxes];
  const nodePlayer = [startPlayer];
  const nodeG = [0];
  const nodeParent = [-1];
  const nodeSeg = [''];
  const bestG = new Map([[keyOf(startBoxes, startLow), 0]]);
  const heapF = [h0];
  const heapId = [0];
  const less = (i, j) => {
    if (heapF[i] !== heapF[j]) return heapF[i] < heapF[j];
    const gi = nodeG[heapId[i]];
    const gj = nodeG[heapId[j]];
    if (gi !== gj) return gi > gj;
    return heapId[i] < heapId[j];
  };
  const heapSwap = (i, j) => {
    const tf = heapF[i]; heapF[i] = heapF[j]; heapF[j] = tf;
    const ti = heapId[i]; heapId[i] = heapId[j]; heapId[j] = ti;
  };
  const heapPush = (f, id) => {
    let i = heapF.length;
    heapF.push(f);
    heapId.push(id);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(i, p)) break;
      heapSwap(i, p);
      i = p;
    }
  };
  const heapPop = () => {
    const topId = heapId[0];
    const lastF = heapF.pop();
    const lastId = heapId.pop();
    if (heapF.length > 0) {
      heapF[0] = lastF;
      heapId[0] = lastId;
      for (let i = 0; ;) {
        let c = 2 * i + 1;
        if (c >= heapF.length) break;
        if (c + 1 < heapF.length && less(c + 1, c)) c++;
        if (!less(c, i)) break;
        heapSwap(i, c);
        i = c;
      }
    }
    return topId;
  };

  let found = -1;
  while (heapF.length > 0) {
    const id = heapPop();
    const boxes = nodeBoxes[id];
    const g = nodeG[id];
    for (const b of boxes) occupied[b] = 1;
    const low = flood(board, occupied, nodePlayer[id], seen, ++stamp, stack);
    const regionStamp = stamp;
    if (bestG.get(keyOf(boxes, low)) !== g) {
      occupied.fill(0);
      continue;
    }
    if (boxes.every((b) => goalSet[b])) {
      found = id;
      occupied.fill(0);
      break;
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const dest = nbr[b * 4 + k];
        if (dest < 0 || wall[dest] || occupied[dest] || !live[dest] || forbidden[dest]) continue;
        const stand = nbr[b * 4 + (k ^ 1)];
        if (stand < 0 || seen[stand] !== regionStamp) continue;
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        occupied[b] = 0;
        occupied[dest] = 1;
        const dead = isDeadlocked(board, occupied, next, dest);
        let nextLow = -1;
        if (!dead) nextLow = flood(board, occupied, b, mark, ++markStamp, stack);
        occupied[dest] = 0;
        occupied[b] = 1;
        if (dead) continue;
        const g2 = g + 1;
        const key = keyOf(next, nextLow);
        const seenG = bestG.get(key);
        if (seenG !== undefined && seenG <= g2) continue;
        const hh = heuristic(board, next);
        if (hh >= INF) continue;
        if (nodeBoxes.length >= maxStates) {
          occupied.fill(0);
          return null;
        }
        const walk = walkPath(board, occupied, nodePlayer[id], stand);
        if (walk === null) continue;
        const nid = nodeBoxes.length;
        nodeBoxes.push(next);
        nodePlayer.push(b);
        nodeG.push(g2);
        nodeParent.push(id);
        nodeSeg.push(walk + PUSH[k]);
        bestG.set(key, g2);
        heapPush(g2 + hh, nid);
      }
    }
    occupied.fill(0);
  }
  if (found < 0) return null;
  const segs = [];
  for (let s = found; s > 0; s = nodeParent[s]) segs.push(nodeSeg[s]);
  segs.reverse();
  return segs.join('');
}

// The square the player ends on after playing `moves` from `from`.
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

// Decompose a level into rooms by cutting out the player's row (hallway).
// Returns [{ squares, boxes, goals, entries }] or null when it does not
// decompose cleanly (a box/goal in the hallway, or a component whose box
// and goal counts differ or are zero-with-goals, or a room with no entry).
function roomsOf(level) {
  const { w, wall, goals, boxes, player, size } = level;
  const hallRow = Math.floor(player / w);
  const comp = new Int32Array(size).fill(-1);
  const comps = [];
  const stack = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || Math.floor(s / w) === hallRow || comp[s] >= 0) continue;
    const squares = [];
    comp[s] = comps.length;
    stack.push(s);
    while (stack.length > 0) {
      const p = stack.pop();
      squares.push(p);
      for (let k = 0; k < 4; k++) {
        const q = nbrOf(level, p, k);
        if (q < 0 || wall[q] || Math.floor(q / w) === hallRow || comp[q] >= 0) continue;
        comp[q] = comps.length;
        stack.push(q);
      }
    }
    comps.push(squares);
  }
  const byComp = comps.map(() => ({ boxes: [], goals: [] }));
  for (const b of boxes) {
    if (comp[b] < 0) return null;
    byComp[comp[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (comp[g] < 0) return null;
    byComp[comp[g]].goals.push(g);
  }
  const rooms = [];
  for (let ci = 0; ci < comps.length; ci++) {
    const cb = byComp[ci].boxes;
    const cg = byComp[ci].goals;
    if (cb.length === 0 && cg.length === 0) continue;
    if (cb.length === 0 || cb.length !== cg.length) return null;
    const entries = new Set();
    for (const s of comps[ci]) {
      for (let k = 0; k < 4; k++) {
        const q = nbrOf(level, s, k);
        if (q >= 0 && !wall[q] && Math.floor(q / w) === hallRow) entries.add(q);
      }
    }
    if (entries.size === 0) return null;
    rooms.push({
      squares: comps[ci],
      boxes: cb.sort((a, b) => a - b),
      goals: cg.sort((a, b) => a - b),
      entries: [...entries].sort((a, b) => a - b),
    });
  }
  rooms.sort((a, b) => a.squares[0] - b.squares[0]);
  return rooms;
}

function nbrOf(level, p, k) {
  const { w, size } = level;
  const x = p % w;
  if (k === 2 && x === 0) return -1;
  if (k === 3 && x === w - 1) return -1;
  const q = p + DY[k] * w + DX[k];
  return q < 0 || q >= size ? -1 : q;
}

function solveLevel(text) {
  const level = parseLevel(text);
  const { w, h, wall, goals, boxes, player, size } = level;
  const hallRow = Math.floor(player / w);
  const noForbidden = new Uint8Array(size);
  const rooms = roomsOf(level);
  if (rooms !== null && rooms.length > 0) {
    const full = makeBoard(w, h, wall, goals);
    const curBoxes = new Set(boxes);
    let playerPos = player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      // Sub-board: the room's squares plus the hallway row (walkable).
      const subWall = new Uint8Array(size).fill(1);
      for (const s of room.squares) subWall[s] = 0;
      for (let x = 0; x < w; x++) {
        const c = hallRow * w + x;
        if (!wall[c]) subWall[c] = 0;
      }
      // Boxes may never enter the hallway.
      const forbidden = new Uint8Array(size);
      for (let x = 0; x < w; x++) forbidden[hallRow * w + x] = 1;
      const sub = makeBoard(w, h, subWall, room.goals);
      const solved = solveBoard(sub, room.boxes, room.entries[0], forbidden, ROOM_STATES);
      if (solved === null) {
        ok = false;
        break;
      }
      const occupied = new Uint8Array(size);
      for (const b of curBoxes) occupied[b] = 1;
      const walk = walkPath(full, occupied, playerPos, room.entries[0]);
      if (walk === null) {
        ok = false;
        break;
      }
      moves += walk + solved;
      playerPos = endOf(w, playerPos, walk + solved);
      for (const b of room.boxes) curBoxes.delete(b);
      for (const g of room.goals) curBoxes.add(g);
    }
    if (ok) return moves;
  }
  const board = makeBoard(w, h, wall, goals);
  const solved = solveBoard(board, boxes, player, noForbidden, LEVEL_STATES);
  return solved === null ? '' : solved;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const dir = join(INPUT_DIR, id);
  let text;
  try {
    if (!statSync(dir).isDirectory()) continue;
    text = readFileSync(join(dir, 'level.txt'), 'utf8');
  } catch {
    continue;
  }
  let moves = '';
  try {
    moves = solveLevel(text);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
