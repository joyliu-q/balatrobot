import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

// Use the dashboard's installed Svelte, without a second package/install in
// the game repository. Override for checkouts with a different directory name.
const frontend = process.env.TRAINING_GYM_FRONTEND || fileURLToPath(new URL("../../training-gym/dashboards/frontend", import.meta.url));
const require = createRequire(resolve(frontend, "package.json"));
const { compile } = require("svelte/compiler");
const { render } = await import(pathToFileURL(require.resolve("svelte/server")));
const source = readFileSync(new URL("./TrajectoryViewer.svelte", import.meta.url), "utf8");
const compiled = compile(source, { generate: "server", filename: "TrajectoryViewer.svelte" });
const code = compiled.js.code.replace(/from '(svelte[^']*)'/g, (_, id) => `from '${pathToFileURL(require.resolve(id))}'`);
const { default: Viewer } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);
const helpers = new Function(`${source.slice(source.indexOf("  function text("), source.indexOf("  let stateMessages"))}; return { chipProgress, chips, statesFrom, actionsFrom };`)();

const blinds = {
  small: { name: "Small Blind", type: "small", status: "SELECT", score: 300, tag: "Investment Tag", tag_effect: "After defeating the Boss Blind, gain $25." },
  big: { name: "Big Blind", type: "big", status: "UPCOMING", score: 450, tag: "Coupon Tag", tag_effect: "Initial cards and booster packs in the next shop are free." },
  boss: { name: "The Hook", type: "boss", status: "UPCOMING", score: 600, effect: "Discards 2 random cards per hand played." },
};
const initial = { phase: "BLIND_SELECT", ante: 1, round: 0, blinds };
const stateMessage = (state) => ({ role: "tool", content: `<balatro_state>${JSON.stringify(state)}</balatro_state>` });
const htmlFor = (state, action = null) => render(Viewer, { props: {
  trajectory: [...(action ? [{ role: "assistant", content: JSON.stringify(action) }] : []), stateMessage(state)],
} }).body;

test("both Svelte targets compile without warnings", () => {
  assert.deepEqual(compiled.warnings, []);
  assert.deepEqual(compile(source, { generate: "client" }).warnings, []);
});

test("initial blind selection is visible before the first action", () => {
  const html = render(Viewer, { props: { trajectory: [{ role: "user", content: `Gamestate: ${JSON.stringify(initial)}` }] } }).body;
  for (const label of ["Initial state", "AVAILABLE NOW", "300 chips", "450 chips", "600 chips", "Investment Tag", "gain $25", "Coupon Tag", "Cannot skip the Boss Blind"])
    assert.ok(html.includes(label), label);
  assert.ok(!html.includes("No structured Balatro state"));
});

test("hand displays explicit scored, required and remaining chips", () => {
  const html = htmlFor({ phase: "hand", blind: { type: "big", score: [1200, 2400] } });
  for (const label of ["CHIPS SCORED", "CHIPS REQUIRED TO CLEAR", "CHIPS STILL NEEDED", "1,200", "2,400"])
    assert.ok(html.includes(label), label);
});

test("selection resets displayed score instead of inheriting the last hand", () => {
  assert.deepEqual(helpers.chipProgress({ ...initial, blind: { score: [999, 300] } }), {
    current: 0, target: 300, label: "Small Blind", pending: true,
  });
});

test("shop and pack never display stale active-blind chips", () => {
  for (const phase of ["shop", "pack"]) {
    assert.equal(helpers.chipProgress({ phase, blind: { score: [999, 300] } }), null);
    const html = htmlFor({ phase, blind: { score: [999, 300] } });
    assert.ok(html.includes("No active blind score"));
    assert.ok(!html.includes("CHIPS SCORED"));
  }
});

test("zero and missing scores stay distinct; overshoot needs zero more chips", () => {
  assert.equal(helpers.chips(0), "0");
  for (const value of [null, undefined, "", "not recorded", Infinity]) assert.equal(helpers.chips(value), "—");
  const html = htmlFor({ phase: "hand", blind: { score: [650, 600] } });
  assert.match(html, /CHIPS STILL NEEDED<\/small>\s*<strong[^>]*>0<\/strong>/);
});

test("missing tag effects are acknowledged, not invented", () => {
  const html = htmlFor({ ...initial, blinds: { small: { ...blinds.small, tag_effect: null } } });
  assert.ok(html.includes("Effect not recorded in this trajectory."));
});

test("skip outcome shows the earned tag separately from the next offer", () => {
  const html = htmlFor({ ...initial,
    blinds: { ...blinds, small: { ...blinds.small, status: "SKIPPED" }, big: { ...blinds.big, status: "SELECT" } },
    action_result: { method: "skip", skipped: { name: "Small Blind" }, tag: blinds.small.tag, tag_effect: blinds.small.tag_effect, next: blinds.big },
  }, { method: "skip", params: {} });
  for (const label of ["TAG TAKEN · IN-GAME SKIP REWARD", "Investment Tag", "gain $25", "Coupon Tag", "450 chips required"])
    assert.ok(html.includes(label), label);
});

test("a failed skip does not claim a tag was earned", () => {
  const html = htmlFor(initial, { method: "skip", params: {} });
  assert.ok(!html.includes("TAG TAKEN · IN-GAME SKIP REWARD"));
});

test("initial snapshot seeds sparse deltas without shifting actions", () => {
  const messages = [
    { role: "assistant", content: '{"method":"select","params":{}}' },
    stateMessage({ phase: "hand", blind: { type: "small", score: [0, 300] } }),
    stateMessage({ blind: null, phase: "shop" }),
  ];
  const states = helpers.statesFrom(messages, initial);
  assert.equal(states.length, 2);
  assert.equal(states[0].ante, 1);
  assert.equal(states[0].blinds.small.tag, "Investment Tag");
  assert.equal(states[1].blind, undefined);
  assert.deepEqual(helpers.actionsFrom(messages), [{ method: "select", params: {} }]);
});
