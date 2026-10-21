// Run with: node --test tests/test_navigation_button.mjs
// Execute the page's real JavaScript with simulated DOM and HTTP boundaries.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import vm from "node:vm";

class Element {
  constructor() {
    this.disabled = false;
    this.hidden = false;
    this.textContent = "";
    this.style = {};
    this.attributes = {};
    this.listeners = {};
    this.classList = { toggle: ignore };
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  append() {}
  appendChild() {}
  blur() {}
}

class Response {
  constructor(data, status = 200) {
    this.data = data;
    this.status = status;
    this.ok = status >= 200 && status < 300;
  }
  async json() { return structuredClone(this.data); }
}

class PageHarness {
  constructor() {
    this.nodes = new Map();
    this.events = {};
    this.requests = [];
    this.stopResponse = new Response({ status: "stopping", command_id: "voice-1" });
    this.stopDeferred = null;
    this.statusError = false;
    this.snapshot = {
      state: "executing", max_record_seconds: 30, vln_enabled: true,
      model: "small.en", device: "cpu", compute_type: "int8", matching: true,
      vln_url: "http://127.0.0.1:8000", server_time: 100, history: [],
      task: { command_id: "voice-1", status: "running", started: 90, ended: null },
    };
    const html = readFileSync(new URL("../stt/web/index.html", import.meta.url), "utf8");
    for (const match of html.matchAll(/id="([^"]+)"/g)) this.nodes.set(match[1], new Element());
    this.context = vm.createContext({
      document: {
        getElementById: this.getElement.bind(this), createElement: createElement,
        addEventListener: this.addEventListener.bind(this),
      },
      fetch: this.fetch.bind(this), setInterval: interval, clearInterval: ignore,
    });
    vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], this.context);
  }
  getElement(id) { return this.nodes.get(id) || null; }
  addEventListener(name, handler) { this.events[name] = handler; }
  async fetch(path, options) {
    this.requests.push({ path, method: options.method });
    if (path === "/api/status") {
      if (this.statusError) throw new Error("Offline");
      return new Response(this.snapshot);
    }
    assert.equal(path, "/api/navigation/stop");
    return this.stopDeferred ? this.stopDeferred.promise : this.stopResponse;
  }
  clickStop() { return this.getElement("stop-navigation").listeners.click(); }
  refresh() { return vm.runInContext("refresh()", this.context); }
  stopRequests() { return this.requests.filter(isStopRequest); }
}

function ignore() {}
function interval() { return 1; }
function createElement() { return new Element(); }
function isStopRequest(request) { return request.path === "/api/navigation/stop"; }
async function page() {
  const harness = new PageHarness();
  await setImmediate();
  return harness;
}

async function buttonAvailability() {
  const h = await page();
  assert.equal(h.getElement("stop-navigation").disabled, false);
  for (const state of ["idle", "listening", "processing"]) {
    h.snapshot.state = state;
    await h.refresh();
    assert.equal(h.getElement("stop-navigation").disabled, true);
  }
  h.snapshot.state = "executing";
  h.snapshot.vln_enabled = false;
  await h.refresh();
  assert.equal(h.getElement("stop-navigation").disabled, true);
  h.snapshot.vln_enabled = true;
  h.statusError = true;
  await h.refresh();
  assert.equal(h.getElement("stop-navigation").disabled, true);
}

async function singleRequestAndCleanupLock() {
  const h = await page();
  h.stopDeferred = Promise.withResolvers();
  const first = h.clickStop();
  assert.equal(h.getElement("stop-navigation").textContent, "Requesting stop…");
  assert.equal(h.getElement("stop-navigation").disabled, true);
  await h.clickStop();
  assert.equal(h.stopRequests().length, 1);
  assert.equal(h.stopRequests()[0].method, "POST");
  h.stopDeferred.resolve(h.stopResponse);
  await first;
  assert.equal(h.getElement("stop-navigation").textContent, "Stopping…");
  assert.equal(h.getElement("talk").disabled, true);
  await h.refresh(); // Old "running" status must not undo the acknowledgement.
  await h.clickStop();
  assert.equal(h.stopRequests().length, 1);
  assert.equal(h.getElement("stop-navigation").disabled, true);
}

async function errorAllowsRetry() {
  const h = await page();
  h.stopResponse = new Response({ error: "Could not confirm navigation stop. Please retry." }, 502);
  await h.clickStop();
  assert.match(h.getElement("error").textContent, /Please retry/);
  assert.equal(h.getElement("stop-navigation").disabled, false);
  assert.equal(h.getElement("talk").disabled, true);
  h.stopResponse = new Response({ status: "stopping", command_id: "voice-1" });
  await h.clickStop();
  assert.equal(h.stopRequests().length, 2);
  assert.equal(h.getElement("error").textContent, "");
  assert.equal(h.getElement("stop-navigation").textContent, "Stopping…");
}

async function invalidAcknowledgementAllowsRetry() {
  for (const reply of [{}, { status: "idle" }, { status: "stopping", command_id: "other" }]) {
    const h = await page();
    h.stopResponse = new Response(reply);
    await h.clickStop();
    assert.notEqual(h.getElement("error").textContent, "");
    assert.equal(h.getElement("stop-navigation").disabled, false);
  }
}

async function nextTaskGetsFreshButton() {
  const h = await page();
  await h.clickStop();
  h.snapshot.state = "idle";
  h.snapshot.task.status = "cancelled";
  await h.refresh();
  assert.equal(h.getElement("stop-navigation").textContent, "Stop Navigation");
  assert.equal(h.getElement("stop-navigation").disabled, true);
  h.snapshot.state = "executing";
  h.snapshot.task = { command_id: "voice-2", status: "running", started: 100, ended: null };
  await h.refresh();
  assert.equal(h.getElement("stop-navigation").disabled, false);
}

async function stoppingThenCancelledRendersAndUnlocks() {
  const h = await page();
  h.nodes.set("task", new Element());
  vm.runInContext("shownId = 'voice-1'", h.context);
  h.snapshot.task.status = "stopping";
  await h.refresh();
  assert.equal(h.getElement("pill-text").textContent, "Stopping navigation");
  assert.equal(h.getElement("talk").disabled, true);
  assert.equal(h.getElement("stop-navigation").textContent, "Stopping…");
  assert.match(h.getElement("task").textContent, /Waiting for cleanup/);
  h.snapshot.state = "idle";
  h.snapshot.task.status = "cancelled";
  h.snapshot.task.ended = 110;
  await h.refresh();
  assert.equal(h.getElement("pill-text").textContent, "Ready");
  assert.equal(h.getElement("talk").disabled, false);
  assert.equal(h.getElement("task").textContent, "Navigation stopped.");
  assert.equal(h.getElement("stop-navigation").disabled, true);
}

async function cleanupFailureRendersAndUnlocks() {
  const h = await page();
  h.nodes.set("task", new Element());
  vm.runInContext("shownId = 'voice-1'", h.context);
  h.snapshot.state = "idle";
  h.snapshot.task.status = "failed";
  h.snapshot.task.message = "RuntimeError: Robot stop failed";
  h.snapshot.task.ended = 110;
  await h.refresh();
  assert.equal(h.getElement("talk").disabled, false);
  assert.match(h.getElement("task").textContent, /Robot stop failed/);
}

async function keyboardDoesNotHijackFocusedButton() {
  const h = await page();
  const focusedButton = h.getElement("stop-navigation");
  focusedButton.closest = returnSelf;
  for (const type of ["keydown", "keyup"]) {
    h.events[type]({ type, code: "Space", repeat: false, target: focusedButton,
      preventDefault: failPreventDefault });
  }
  assert.equal(h.stopRequests().length, 0); // Native button activation handles Space.
}

function returnSelf() { return this; }
function failPreventDefault() { assert.fail("Focused button's Space key must retain native behavior"); }

test("stop is enabled only during navigation", buttonAvailability);
test("one POST per click sequence; acknowledgement keeps cleanup locked", singleRequestAndCleanupLock);
test("request errors leave listening locked and permit retry", errorAllowsRetry);
test("invalid or mismatched replies do not claim stopping", invalidAcknowledgementAllowsRetry);
test("a subsequent task gets a fresh stop button", nextTaskGetsFreshButton);
test("stopping stays locked and cancelled unlocks listening", stoppingThenCancelledRendersAndUnlocks);
test("cleanup failure is shown and unlocks listening", cleanupFailureRendersAndUnlocks);
test("Space shortcut does not hijack a focused button", keyboardDoesNotHijackFocusedButton);
