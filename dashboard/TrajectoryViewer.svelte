<script>
  /** Balatro trajectory viewer shipped with balatrobot.
   * Props follow the Training Gym trajectory-viewer contract. */
  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
  } = $props();

  function text(value) {
    return typeof value === "string" ? value : value == null ? "" : String(value);
  }

  function area(state, name) {
    const value = state?.[name];
    if (Array.isArray(value)) return value;
    if (Array.isArray(value?.items)) return value.items;
    if (value && typeof value === "object") return Object.entries(value).map(([id, effect]) => ({ id, effect }));
    return [];
  }

  function merge(previous, next) {
    if (!next || typeof next !== "object" || Array.isArray(next)) return next;
    const result = { ...(previous || {}) };
    for (const [key, value] of Object.entries(next)) {
      if (value === null) delete result[key];
      else if (value && typeof value === "object" && !Array.isArray(value)) result[key] = merge(result[key], value);
      else result[key] = value;
    }
    return result;
  }

  function statesFrom(messages, initial = null) {
    const output = [];
    for (const message of Array.isArray(messages) ? messages : []) {
      const matches = text(message?.content).matchAll(/<balatro_state>([\s\S]*?)<\/balatro_state>/g);
      for (const match of matches) {
        try {
          const payload = JSON.parse(match[1]);
          output.push(payload?.state && typeof payload.state === "object" ? payload.state : payload);
        } catch {
          // A malformed state should not prevent the rest of the run rendering.
        }
      }
    }
    return output.reduce((all, state) => [...all, merge(all.at(-1) || initial, state)], []);
  }

  function jsonObjectEnd(value, start) {
    let depth = 0;
    let quoted = false;
    let escaped = false;
    for (let index = start; index < value.length; index += 1) {
      const char = value[index];
      if (quoted) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === '"') quoted = false;
      } else if (char === '"') quoted = true;
      else if (char === "{") depth += 1;
      else if (char === "}" && --depth === 0) return index + 1;
    }
    return -1;
  }

  function actionsFrom(messages) {
    const actions = [];
    for (const message of Array.isArray(messages) ? messages : []) {
      // Tool results can contain a fallback action inside the structured
      // ``<balatro_state>`` payload.  That is not a model action and must not
      // shift the action/state alignment used by the turn viewer.
      const content = text(message?.content).replace(/<balatro_state>[\s\S]*?<\/balatro_state>/g, "");
      let cursor = 0;
      while (cursor < content.length) {
        const match = /\{"method"\s*:/.exec(content.slice(cursor));
        if (!match) break;
        const start = cursor + match.index;
        const end = jsonObjectEnd(content, start);
        if (end < 0) break;
        try {
          const action = JSON.parse(content.slice(start, end));
          if (action?.method) actions.push(action);
        } catch {
          // Ignore malformed model output; the state is still useful.
        }
        cursor = end;
      }
    }
    return actions;
  }

  function initialStateFrom(messages) {
    for (const message of Array.isArray(messages) ? messages : []) {
      const content = text(message?.content);
      const marker = content.indexOf("Gamestate:");
      if (marker < 0) continue;
      const start = content.indexOf("{", marker);
      if (start < 0) continue;
      const end = jsonObjectEnd(content, start);
      if (end < 0) continue;
      try {
        const value = JSON.parse(content.slice(start, end));
        if (value && typeof value === "object") return value;
      } catch {
        // Continue searching other messages.
      }
    }
    return null;
  }

  function selectedBlind(state) {
    const blinds = state?.blinds;
    if (!blinds || typeof blinds !== "object" || Array.isArray(blinds)) return null;
    for (const [id, blind] of Object.entries(blinds)) {
      if (blind && String(blind.status || "").toUpperCase() === "SELECT") {
        return { id, ...blind };
      }
    }
    return null;
  }

  function blindLabel(blind) {
    if (!blind) return "—";
    return blind.name || blind.id || blind.type || "blind";
  }

  function number(value) {
    if (value == null || value === "" || typeof value === "boolean") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function chips(value) {
    const parsed = number(value);
    return parsed == null ? "—" : parsed.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }

  function chipProgress(state) {
    const phase = String(state?.phase || "").toUpperCase();
    // Scores belong to a blind, not the whole run. Never carry the previous
    // hand's chips into a shop, pack, or a newly offered blind.
    if (phase === "BLIND_SELECT") {
      const blind = selectedBlind(state);
      return blind ? { current: 0, target: number(blind.score), label: blindLabel(blind), pending: true } : null;
    }
    if (!["HAND", "SELECTING_HAND", "GAME_OVER"].includes(phase)) return null;
    const score = state?.blind?.score;
    if (!Array.isArray(score)) return null;
    return { current: number(score[0]), target: number(score[1]), label: blindLabel(state.blind), pending: false };
  }

  function cardInfo(card) {
    const id = String(typeof card === "string" ? card : card?.id || card?.key || card?.label || "?");
    const modifier = typeof card === "object" ? card?.mod || card?.modifier || null : null;
    const match = id.match(/^([SHDC])_(.+)$/);
    if (!match) return { id, rank: id, suit: "", red: false, modifier };
    return {
      id,
      rank: match[2] === "T" ? "10" : match[2],
      suit: ({ S: "♠", H: "♥", D: "♦", C: "♣" })[match[1]],
      red: match[1] === "H" || match[1] === "D",
      modifier,
    };
  }

  function modifierValue(card, key) {
    const value = (card?.mod || card?.modifier)?.[key];
    return typeof value === "string" ? value.toLowerCase().replaceAll(" ", "_") : "";
  }

  function prettyModifier(value) {
    return String(value).replaceAll("_", " ").toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function modifierBadges(card) {
    const modifiers = card?.mod || card?.modifier || {};
    const badges = [];
    for (const [kind, value] of Object.entries(modifiers)) {
      if (value == null || value === false || value === "" || String(value).toUpperCase() === "BASE") continue;
      const label = kind === "seal" ? `${prettyModifier(value)} Seal`
        : kind === "perishable" ? `Perishable · ${value} rounds`
        : value === true ? prettyModifier(kind)
        : ["edition", "enhancement"].includes(kind) ? prettyModifier(value)
        : `${prettyModifier(kind)} · ${prettyModifier(value)}`;
      badges.push({ kind, value: String(value).toLowerCase(), label });
    }
    if (card?.debuff || card?.state?.debuff) badges.push({ kind: "debuff", value: "true", label: "Debuffed" });
    return badges;
  }

  function itemName(item) {
    const id = String(typeof item === "string" ? item : item?.id || item?.key || item?.label || "item");
    return id.replace(/^[jcv]_/, "").split(/[_-]+/).filter(Boolean)
      .map((part) => part[0].toUpperCase() + part.slice(1)).join(" ") || id;
  }

  function cardTitle(card) {
    const info = cardInfo(card);
    return [info.rank + info.suit, ...modifierBadges(card).map((badge) => badge.label)].join(" · ");
  }

  let stateMessages = $derived(
    text(sample?.response).includes("<balatro_state>")
      ? [{ role: "assistant", content: sample.response }]
      : trajectory,
  );
  let initialState = $derived(
    initialStateFrom(trajectory)
      || initialStateFrom(Array.isArray(sample?.prompt) ? sample.prompt : [{ content: sample?.prompt }])
      || initialStateFrom(stateMessages),
  );
  let states = $derived(statesFrom(stateMessages, initialState));
  let actions = $derived(actionsFrom(stateMessages));
  let groups = $derived.by(() => {
    const result = [];
    const frames = states.map((state, index) => ({ state, index }));
    if (initialState) frames.unshift({ state: initialState, index: -1 });
    for (const { index, state } of frames) {
      // SHOP/PACK/BLIND_SELECT deliberately have no active blind object.
      // Keep those phase transitions as visible tabs instead of collapsing
      // them into an indistinguishable generic "blind" group.
      const blind = state?.blind?.type || state?.blind?.name || state?.phase || "blind";
      const key = `${state?.ante ?? "?"}:${state?.round ?? "?"}:${blind}`;
      let group = result.at(-1);
      if (!group || group.key !== key) {
        group = { key, ante: state?.ante, round: state?.round, blind, frames: [] };
        result.push(group);
      }
      group.frames.push({ state, index });
    }
    return result;
  });
  let activeGroupIndex = $state(0);
  let activeFrameIndex = $state(0);
  let activeGroup = $derived(groups[activeGroupIndex] || groups.at(-1) || null);
  let activeFrame = $derived(activeGroup?.frames[Math.min(activeFrameIndex, Math.max(activeGroup.frames.length - 1, 0))] || null);
  let latest = $derived(activeFrame?.state || states.at(-1) || {});
  let progress = $derived(chipProgress(latest));
  let chipsRemaining = $derived(
    progress?.target != null && progress?.current != null ? Math.max(0, progress.target - progress.current) : null,
  );
  let blindChoices = $derived(
    String(latest.phase || "").toUpperCase() === "BLIND_SELECT"
      ? Object.entries(latest.blinds || {}).map(([id, blind]) => ({ id, ...blind })) : [],
  );
  let activeAction = $derived(actions[activeFrame?.index ?? -1] || null);
  let previousState = $derived(activeFrame?.index > 0 ? states[activeFrame.index - 1] : initialState);
  let selectedIndices = $derived(
    activeAction && ["play", "discard", "use"].includes(activeAction.method) && Array.isArray(activeAction.params?.cards)
      ? activeAction.params.cards
      : [],
  );
  // A frame is the state *after* its corresponding action.  In particular,
  // the final play of a blind can legitimately result in SHOP, where the
  // game has no hand.  Keep the input cards visible for that action, but do
  // not present them as cards still held in the resulting state.
  let actionUsesInputHand = $derived(
    selectedIndices.length > 0 && previousState && ["play", "discard", "use"].includes(activeAction?.method),
  );
  let handForAction = $derived(actionUsesInputHand ? area(previousState, "hand") : hand);
  let handLabel = $derived(
    actionUsesInputHand
      ? `${activeAction.method.toUpperCase()} INPUT · PRE-${String(latest.phase || "HAND").toUpperCase()} STATE`
      : latest.phase === "shop" ? "HAND · NO HAND IN SHOP"
      : "HAND",
  );
  // Balatro advances ante_num immediately after the boss is beaten, before
  // the post-boss shop opens.  Calling this out avoids making the correct
  // next-ante value look like an off-by-one error next to the prior round.
  let enteredNewAnte = $derived(
    latest.phase === "shop"
      && Number.isFinite(Number(previousState?.ante))
      && Number(latest.ante) === Number(previousState.ante) + 1,
  );
  let anteLabel = $derived(
    latest.ante != null ? `ANTE ${latest.ante}${enteredNewAnte ? " · NEW ANTE (BOSS CLEARED)" : ""}` : "",
  );
  let actionTarget = $derived(
    activeAction?.method === "use" ? activeAction.params?.consumable
      : activeAction?.method === "buy" ? activeAction.params?.card ?? activeAction.params?.voucher ?? activeAction.params?.pack
      : activeAction?.method === "sell" ? activeAction.params?.joker ?? activeAction.params?.consumable
      : null,
  );
  let actionTargetName = $derived(
    activeAction?.method === "use" ? "consumable"
      : activeAction?.method === "buy" ? (activeAction.params?.voucher != null ? "voucher" : activeAction.params?.pack != null ? "pack" : "card")
      : activeAction?.method === "sell" ? (activeAction.params?.joker != null ? "joker" : "consumable")
      : "",
  );
  let actionCards = $derived(
    activeAction && Array.isArray(activeAction.params?.cards) ? activeAction.params.cards : [],
  );
  let actionConsumable = $derived(
    activeAction?.method === "use" && Number.isInteger(actionTarget) && previousState
      ? area(previousState, "consumables")[actionTarget] || null
      : null,
  );
  let hand = $derived(area(latest, "hand"));
  let deck = $derived(area(latest, "deck"));
  let discard = $derived(area(latest, "discard"));
  let deckCount = $derived(latest.deck_left ?? (deck.length > 0 ? deck.length : null));
  let discardCount = $derived(
    latest.discard_count ?? latest.discards ?? latest.blind?.discards ?? (discard.length > 0 ? discard.length : null),
  );
  let jokers = $derived(area(latest, "jokers"));
  let consumables = $derived(area(latest, "consumables"));
  let consumablesForView = $derived(
    activeAction?.method === "use" && actionConsumable
      ? area(previousState, "consumables")
      : consumables,
  );
  let vouchers = $derived(area(latest, "vouchers"));
  let ownedVouchers = $derived(area(latest, "owned_vouchers"));
  let handValues = $derived(
    latest.hand_values && typeof latest.hand_values === "object" ? latest.hand_values : {},
  );
  let latestReward = $derived(Array.isArray(rewardEvents) ? rewardEvents.at(-1) : null);
  let skipResult = $derived.by(() => {
    if (activeAction?.method !== "skip") return null;
    // New harnesses attach this directly to the resulting state delta.
    if (latest?.action_result?.method === "skip") return latest.action_result;
    // Fall back to deriving it for older trajectories.
    const skipped = previousState ? Object.entries(previousState.blinds || {})
      .map(([id, blind]) => ({ id, ...(blind || {}) }))
      .find((blind) => String(blind.status || "").toUpperCase() === "SELECT") : null;
    const next = selectedBlind(latest);
    if (!skipped || String(latest.blinds?.[skipped.id]?.status || "").toUpperCase() !== "SKIPPED") return null;
    return {
      method: "skip",
      skipped: { id: skipped.id, name: blindLabel(skipped), type: skipped.type },
      tag: skipped.tag,
      tag_effect: skipped.tag_effect,
      next: next ? { id: next.id, name: blindLabel(next), type: next.type, score: next.score, effect: next.effect } : null,
    };
  });

  function selectGroup(index) {
    activeGroupIndex = index;
    activeFrameIndex = 0;
  }
</script>

{#snippet finish(card)}
  {#if modifierValue(card, "edition") && modifierValue(card, "edition") !== "base"}
    <span class="finish-layer" aria-hidden="true"></span>
  {/if}
{/snippet}

{#snippet badges(card)}
  {@const modifiers = modifierBadges(card)}
  {#if modifiers.length}
    <div class="modifier-badges" aria-label="Card modifiers">
      {#each modifiers as badge}
        <span class="modifier-badge" data-kind={badge.kind} data-value={badge.value}>{badge.label}</span>
      {/each}
    </div>
  {/if}
{/snippet}

<section class="balatro-viewer" data-trajectory-viewer="balatro">
  <header class="viewer-header">
    <div>
      <div class="eyebrow">BALATRO · TRAJECTORY</div>
      <h3>Run board</h3>
      <p>{groups.length} blinds · {states.length} turns · {Array.isArray(samples) ? samples.length : 0} samples</p>
    </div>
    <div class="run-badge">{latest.phase || run?.status || "RUN"}<small>{latest.ante != null ? `ANTE ${latest.ante}` : ""}</small></div>
  </header>

  {#if groups.length}
    <nav class="blind-strip" aria-label="Ante, round, and blind">
      {#each groups as group, index}
        <button class:active={index === activeGroupIndex} class="blind-tab" onclick={() => selectGroup(index)}>
          <span class="tab-index">{String(index + 1).padStart(2, "0")}</span>
          <span><b>{group.blind}</b><small>{group.ante != null ? `ante ${group.ante}` : ""}{group.round != null ? ` · blind ${group.round}` : ""}</small></span>
        </button>
      {/each}
    </nav>

    {#if activeGroup}
      <div class="turn-strip" aria-label="Turns in selected blind">
        <span>{activeGroup.frames.length} turns</span>
        {#each activeGroup.frames as frame, index}
          <button class:active={index === activeFrameIndex} class="turn-dot" title={frame.index < 0 ? "Initial state · before any action" : `turn ${frame.index + 1}`} onclick={() => (activeFrameIndex = index)}>{frame.index < 0 ? "S" : frame.index + 1}</button>
        {/each}
      </div>

      {#if activeAction}
        <div class:action-play={activeAction.method === "play"} class:action-discard={activeAction.method === "discard"} class:action-use={activeAction.method === "use"} class="action-banner">
          <span class="action-method">{activeAction.method.replaceAll("_", " ")}</span>
          {#if actionCards.length}<span>cards [{actionCards.join(", ")}]</span>{/if}
          {#if actionTarget != null}<span>{actionTargetName} {actionTarget}{#if actionConsumable} · {itemName(actionConsumable)}{/if}</span>{/if}
          {#if activeAction.method === "skip"}<span>skip blind · take its tag</span>{/if}
          {#if activeAction.method === "next_round"}<span>leave shop · choose next blind</span>{/if}
        </div>
        {#if skipResult}
          <div class="skip-result">
            <div class="skip-result-title"><span>TAG TAKEN · IN-GAME SKIP REWARD</span><b>{skipResult.tag || "Tag not recorded"}</b></div>
            <div class="skip-result-grid">
              <div><small>SKIPPED</small><strong>{blindLabel(skipResult.skipped)}</strong></div>
              <div><small>EFFECT / WHEN IT APPLIES</small><strong>{skipResult.tag_effect || "Effect not recorded in this trajectory."}</strong></div>
              {#if skipResult.next}<div><small>NEXT BLIND</small><strong>{blindLabel(skipResult.next)}{#if skipResult.next.score != null} · {chips(skipResult.next.score)} chips required{/if}</strong></div>{/if}
            </div>
          </div>
        {/if}
      {/if}

      <div class="felt">
        <div class="felt-topline">
          <div><span class="pip"></span><b>{String(progress?.label || latest.phase || "BLIND").replaceAll("_", " ")}</b><small>{anteLabel}{latest.round != null ? ` · BLIND ${latest.round}` : ""}</small></div>
        </div>

        {#if progress}
          <div class="chip-panel" aria-label="Blind chip progress">
            <div><small>CHIPS SCORED · THIS BLIND</small><strong>{chips(progress.current)}</strong></div>
            <div><small>CHIPS REQUIRED TO CLEAR</small><strong>{chips(progress.target)}</strong></div>
            <div><small>CHIPS STILL NEEDED</small><strong>{chips(chipsRemaining)}</strong></div>
          </div>
          <p class="chip-note">{progress.pending ? "Blind not started. Chips start at 0 when you select it." : "Chips are cumulative within this blind and reset for the next blind."} Each ante has separate Small, Big, and Boss targets.</p>
        {:else}
          <p class="chip-note">No active blind score in this phase. Chips reset when the next blind starts.</p>
        {/if}

        {#if blindChoices.length}
          <div class="section-label"><span>BLIND SELECTION · TARGETS & SKIP REWARDS</span></div>
          <div class="blind-choices">
            {#each blindChoices as blind}
              {@const status = String(blind.status || "UNKNOWN").toUpperCase()}
              {@const boss = String(blind.type || blind.id).toUpperCase() === "BOSS"}
              <article class:offered={status === "SELECT"} class="blind-choice">
                <div class="choice-heading"><b>{blindLabel(blind)}</b><small>{status === "SELECT" ? "AVAILABLE NOW" : status}</small></div>
                <p class="choice-target">Clear target: <strong>{chips(blind.score)} chips</strong></p>
                {#if blind.effect}<p>{blind.effect}</p>{/if}
                {#if boss}
                  <div class="tag-offer"><b>Cannot skip the Boss Blind</b><p>No skip reward.</p></div>
                {:else}
                  <div class="tag-offer"><small>{status === "SKIPPED" ? "TAG TAKEN" : status === "DEFEATED" ? "SKIP OFFER · NOT TAKEN" : "SKIP TO RECEIVE"}</small><b>{blind.tag || "Tag not recorded"}</b><p>{blind.tag_effect || "Effect not recorded in this trajectory."}</p></div>
                {/if}
              </article>
            {/each}
          </div>
          <p class="chip-note">Skipping takes the tag instead of playing that blind. The description above states when its benefit applies; this is an in-game reward, not the training reward.</p>
        {/if}

        <div class="table-zone">
          <div class="pile deck-pile"><strong>{deckCount ?? "—"}</strong><small>DECK</small></div>
          <div class="table-center"><span>{latest.phase || "HAND"}</span>{#if latest.money != null}<strong>${latest.money}</strong>{/if}</div>
          <div class="pile discard-pile"><strong>{discardCount ?? "—"}</strong><small>DISCARD</small></div>
        </div>

        <div class="section-label"><span>{handLabel}</span><small>{handForAction.length} cards{latest.blind?.hands != null ? ` · ${latest.blind.hands} hands left` : ""}</small></div>
        {#if selectedIndices.length}
          <div class="selection-note"><span class="selection-dot"></span>{activeAction.method} selected cards <b>[{selectedIndices.join(", ")}]</b></div>
        {/if}
        {#if handForAction.length}
          <div class="hand">
            {#each handForAction as card, index (`${activeFrame?.index}-${index}-${cardTitle(card)}`)}
              {@const info = cardInfo(card)}
              {@const isSelected = selectedIndices.includes(index)}
              <div class="card-slot" class:dimmed={selectedIndices.length > 0 && !isSelected}>
                <div class:red={info.red} class:selected={isSelected} class="playing-card modifier-surface" data-edition={modifierValue(card, "edition")} data-enhancement={modifierValue(card, "enhancement")} title={cardTitle(card)}>
                  {@render finish(card)}
                  <span class="corner"><b>{info.rank}</b><i>{info.suit}</i></span><strong>{info.suit || "·"}</strong>
                  {#if modifierValue(card, "seal")}<span class="wax-seal" data-seal={modifierValue(card, "seal")} aria-label={`${prettyModifier(modifierValue(card, "seal"))} Seal`}>✦</span>{/if}
                  {#if isSelected}<span class="selected-mark">✓</span>{/if}
                </div>
                {@render badges(card)}
              </div>
            {/each}
          </div>
        {:else}<div class="empty-section">No hand data at this turn.</div>{/if}

        {#if Object.keys(handValues).length}
          <div class="section-label scoring-label"><span>POKER HAND VALUES</span><small>BASE CHIPS × MULT</small></div>
          <div class="hand-values">
            {#each Object.entries(handValues) as [name, values]}
              {@const level = Array.isArray(values) ? values[0] : null}
              {@const chips = Array.isArray(values) ? values[1] : null}
              {@const mult = Array.isArray(values) ? values[2] : null}
              <div class:active={latest.hand_stats?.[name]?.[2] > 0} class="hand-value">
                <span>{name}{#if level != null && level > 1}<small>lvl {level}</small>{/if}</span>
                <b>{chips ?? "—"} × {mult ?? "—"} <em>=</em> {chips != null && mult != null ? chips * mult : "—"}</b>
              </div>
            {/each}
          </div>
        {/if}

        {#if jokers.length || consumablesForView.length || vouchers.length || ownedVouchers.length}
          <div class="inventory">
            {#if jokers.length}<div><div class="section-label"><span>JOKERS</span></div><div class="items">{#each jokers as joker}<article class="item joker modifier-surface" data-edition={modifierValue(joker, "edition")}>
              {@render finish(joker)}<b>★ {itemName(joker)}</b>{@render badges(joker)}
              {#if joker.effect}<p>{joker.effect}</p>{/if}{#if joker.sell != null}<small>sell ${joker.sell}</small>{/if}
            </article>{/each}</div></div>{/if}
            {#if consumablesForView.length}<div><div class="section-label"><span>CONSUMABLES</span></div><div class="items">{#each consumablesForView as item, index}<article class:used={activeAction?.method === "use" && actionTarget === index} class="item consumable modifier-surface" data-edition={modifierValue(item, "edition")}>
              {@render finish(item)}<b>✦ {itemName(item)}</b>{@render badges(item)}
              {#if item.effect}<p>{item.effect}</p>{/if}{#if activeAction?.method === "use" && actionTarget === index}<small>used this turn</small>{/if}
            </article>{/each}</div></div>{/if}
            {#if vouchers.length || ownedVouchers.length}<div><div class="section-label"><span>VOUCHERS</span></div><div class="items">{#each vouchers as item}<article class="item voucher"><b>◆ {itemName(item)}</b>{#if item.effect}<p>{item.effect}</p>{/if}{#if item.cost != null}<small>cost ${item.cost}</small>{/if}</article>{/each}{#each ownedVouchers as item}<article class="item voucher owned"><b>◆ {itemName(item)}</b>{#if item.effect}<p>{item.effect}</p>{/if}<small>owned</small></article>{/each}</div></div>{/if}
          </div>
        {/if}
      </div>
      <footer>{latestReward ? `${latestReward.label || "checkpoint"} · ${Number(latestReward.reward || 0).toFixed(3)} reward` : `turn ${activeFrameIndex + 1} / ${activeGroup.frames.length}`}</footer>
    {/if}
  {:else}<div class="empty-section">No structured Balatro state found.</div>{/if}
</section>

<style>
  .balatro-viewer { --felt: #123b2d; margin: 12px 0 14px; overflow: hidden; border: 1px solid #2d4336; border-radius: 9px; background: #101713; color: #edf5ef; box-shadow: 0 9px 24px rgb(0 0 0 / 20%); }
  .viewer-header { display: flex; justify-content: space-between; gap: 16px; padding: 15px 16px 13px; border-bottom: 1px solid rgb(255 255 255 / 9%); } .eyebrow, .section-label { color: #91b39b; font-size: 9px; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; } h3 { margin: 4px 0 2px; font-size: 18px; } .viewer-header p { margin: 0; color: #789181; font-size: 10px; } .run-badge { display: grid; align-content: start; justify-items: end; color: #f2c866; font: 700 10px ui-monospace, monospace; text-transform: uppercase; } .run-badge small { margin-top: 4px; color: #9ab5a1; font-size: 9px; }
  .blind-strip { display: flex; gap: 5px; overflow-x: auto; padding: 10px 12px; background: #121a15; scrollbar-width: thin; } .blind-tab { display: flex; min-width: 145px; align-items: center; gap: 8px; padding: 7px 9px; border: 1px solid #2e4034; border-radius: 5px; background: #18231c; color: #aabdaf; text-align: left; cursor: pointer; } .blind-tab.active { border-color: #d09b42; background: #2b2519; color: #f5dfae; box-shadow: inset 0 -2px #d09b42; } .tab-index { color: #718278; font: 700 10px ui-monospace, monospace; } .blind-tab b, .blind-tab small { display: block; white-space: nowrap; } .blind-tab b { font-size: 10px; text-transform: uppercase; } .blind-tab small { margin-top: 2px; color: #7f9186; font-size: 9px; }
  .turn-strip { display: flex; align-items: center; gap: 5px; overflow-x: auto; padding: 9px 15px; color: #7f9186; font-size: 9px; } .turn-dot { width: 22px; height: 21px; flex: 0 0 auto; border: 1px solid #3b4940; border-radius: 50%; background: #1c2620; color: #9caf9f; font: 9px ui-monospace, monospace; cursor: pointer; } .turn-dot.active { border-color: #f2b84b; background: #d09b42; color: #17140d; font-weight: 700; }
  .action-banner { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 0 9px 8px; padding: 7px 10px; border: 1px solid #3e4a3f; border-radius: 6px; background: #19221b; color: #c6d5c9; font: 9px ui-monospace, monospace; } .action-banner.action-play { border-color: #765c27; background: #241b0e; } .action-banner.action-discard { border-color: #6b3c38; background: #241313; } .action-banner.action-use { border-color: #4f5f8c; background: #191e31; } .action-method { color: #f2c866; font-weight: 750; text-transform: uppercase; } .action-discard .action-method { color: #e58b83; } .action-use .action-method { color: #b7c9ff; }
  .skip-result { margin: 0 9px 9px; padding: 10px 11px; border: 1px solid #9d7734; border-radius: 6px; background: linear-gradient(135deg, #302514, #1e1a12); color: #f4dfad; } .skip-result-title { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; font: 700 9px ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; } .skip-result-title b { color: #ffe29a; font-size: 11px; letter-spacing: 0; text-align: right; text-transform: none; } .skip-result-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-top: 8px; } .skip-result-grid div { display: grid; gap: 3px; } .skip-result-grid small { color: #ad9c73; font: 8px ui-monospace, monospace; letter-spacing: .08em; } .skip-result-grid strong { color: #f1ead4; font-size: 10px; font-weight: 600; line-height: 1.25; }
  .felt { margin: 0 9px 9px; padding: 14px; border: 1px solid #2a6950; border-radius: 7px; background: radial-gradient(circle at 50% 10%, #1a4b39, var(--felt) 72%); box-shadow: inset 0 0 0 1px rgb(255 255 255 / 4%), inset 0 -18px 40px rgb(0 0 0 / 13%); } .felt-topline { display: flex; justify-content: space-between; align-items: center; gap: 12px; } .felt-topline > div:first-child { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; } .felt-topline small { color: #9fb3a5; font-size: 9px; font-weight: 500; } .pip { width: 8px; height: 8px; border-radius: 50%; background: #e2ad4f; box-shadow: 0 0 0 3px rgb(226 173 79 / 15%); }
  .chip-panel { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 8px; margin-top: 12px; }
  .chip-panel > div { display: grid; gap: 6px; padding: 10px; border: 1px solid #48735a; border-radius: 5px; background: rgb(0 0 0 / 18%); }
  .chip-panel small { color: #b8cebe; font: 9px ui-monospace, monospace; }
  .chip-panel strong { color: #f5e8c9; font: 700 20px ui-monospace, monospace; overflow-wrap: anywhere; }
  .chip-note { margin: 8px 0; color: #adc6b5; font-size: 10px; line-height: 1.5; }
  .blind-choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(195px, 100%), 1fr)); gap: 8px; }
  .blind-choice { padding: 10px; border: 1px solid #47705a; border-radius: 6px; background: rgb(0 0 0 / 18%); }
  .blind-choice.offered { border-color: #e2ad4f; box-shadow: inset 0 2px #e2ad4f; }
  .choice-heading { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px; }
  .choice-heading b { color: #f5e8c9; font-size: 12px; }
  .choice-heading small, .tag-offer small { color: #d9bd83; font: 8px ui-monospace, monospace; }
  .blind-choice p { color: #c5d8cb; font-size: 11px; line-height: 1.45; margin: 6px 0 0; }
  .choice-target strong { color: #fff0ca; }
  .tag-offer { display: grid; gap: 4px; margin-top: 10px; padding-top: 8px; border-top: 1px solid #587158; }
  .tag-offer b { color: #ffe29a; font-size: 11px; }
  .tag-offer p { margin: 0; }
  .table-zone { display: grid; grid-template-columns: 70px 1fr 70px; align-items: center; gap: 12px; min-height: 112px; padding: 8px 0; } .pile { display: grid; place-content: center; width: 54px; height: 70px; justify-self: center; border: 1px solid rgb(255 255 255 / 23%); border-radius: 5px; box-shadow: 3px 4px 0 rgb(0 0 0 / 20%), 6px 7px 0 rgb(0 0 0 / 12%); text-align: center; } .pile strong { color: #f8edcf; font: 700 19px ui-monospace, monospace; } .pile small { color: #b8cdbd; font-size: 8px; letter-spacing: .08em; } .deck-pile { background: repeating-linear-gradient(135deg, #274e41 0 4px, #1b4034 4px 8px); } .discard-pile { background: repeating-linear-gradient(45deg, #5f4840 0 4px, #493631 4px 8px); } .table-center { display: grid; justify-items: center; gap: 7px; color: #b9d2bf; font-size: 9px; letter-spacing: .1em; text-transform: uppercase; } .table-center strong { color: #f2c866; font: 700 20px ui-monospace, monospace; letter-spacing: 0; }
  .hand-values { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 4px 7px; padding: 2px 0 5px; } .hand-value { display: flex; justify-content: space-between; gap: 6px; padding: 5px 7px; border: 1px solid rgb(255 255 255 / 8%); border-radius: 4px; background: rgb(0 0 0 / 12%); color: #a8c2af; font-size: 9px; } .hand-value.active { border-color: #d09b42; color: #f4dfad; } .hand-value b { color: #f0e6ca; font: 700 9px ui-monospace, monospace; white-space: nowrap; } .hand-value em { color: #8da293; font-style: normal; } .hand-value small { margin-left: 3px; color: #d5a74f; font: 8px ui-monospace, monospace; } .scoring-label { margin-top: 4px; }
  .section-label { display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px; padding: 7px 0 5px; border-top: 1px solid rgb(255 255 255 / 10%); } .section-label small { color: #8ca592; font: 9px ui-monospace, monospace; letter-spacing: 0; } .hand { display: flex; justify-content: center; flex-wrap: wrap; gap: 7px; min-height: 92px; } .playing-card { position: relative; width: 59px; height: 84px; overflow: hidden; border: 1px solid #d8d4c7; border-radius: 6px; background: #f0eee5; color: #282e29; box-shadow: 1px 3px 6px rgb(0 0 0 / 30%); transition: transform .12s ease, box-shadow .12s ease; } .playing-card:hover { transform: translateY(-4px); box-shadow: 0 7px 12px rgb(0 0 0 / 35%); } .playing-card.red { color: #a33f3b; } .corner { position: absolute; top: 5px; left: 6px; display: grid; justify-items: center; line-height: 1; } .corner b { font: 700 15px Georgia, serif; } .corner i { font-size: 12px; font-style: normal; } .playing-card > strong { position: absolute; inset: 27px 0 auto; text-align: center; font-size: 27px; line-height: 1; } .playing-card > small { position: absolute; right: 4px; bottom: 4px; left: 4px; overflow: hidden; color: #936c21; font: 700 7px ui-monospace, monospace; text-align: center; text-overflow: ellipsis; text-transform: uppercase; white-space: nowrap; }
  .playing-card.selected { border-color: #f2b84b; box-shadow: 0 0 0 3px rgb(242 184 75 / 32%), 0 8px 15px rgb(0 0 0 / 38%); transform: translateY(-8px); } .playing-card.dimmed { opacity: .48; filter: saturate(.65); } .selected-mark { position: absolute; right: 4px; top: 4px; display: grid; place-items: center; width: 15px; height: 15px; border-radius: 50%; background: #d09b42; color: #17140d; font: 700 10px ui-monospace, monospace; } .selection-note { display: flex; align-items: center; gap: 6px; margin: 2px 0 6px; color: #f1d18d; font-size: 9px; } .selection-note b { color: #f6e7bd; font: 700 9px ui-monospace, monospace; } .selection-dot { width: 6px; height: 6px; border-radius: 50%; background: #f2b84b; box-shadow: 0 0 0 3px rgb(242 184 75 / 16%); }
  .inventory { display: grid; gap: 3px; } .items { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 7px; } .item { min-height: 55px; padding: 8px; border: 1px solid #73532b; border-radius: 6px; background: linear-gradient(145deg, #3d2519, #21150f); } .item.consumable { border-color: #4b587d; background: linear-gradient(145deg, #252b45, #171b2c); } .item.voucher { border-color: #806a35; background: linear-gradient(145deg, #40351c, #211b10); } .item.owned { opacity: .82; } .item.used { border-color: #b7c9ff; box-shadow: 0 0 0 2px rgb(183 201 255 / 25%), 0 5px 12px rgb(0 0 0 / 25%); } .item b { color: #ffe1a1; font-size: 10px; } .item p { margin: 5px 0 0; color: #c2cdbf; font-size: 9px; line-height: 1.3; } .item small { display: block; margin-top: 4px; color: #a5b69f; font: 8px ui-monospace, monospace; text-transform: uppercase; } .empty-section { padding: 22px; color: #9fb3a5; font-size: 10px; text-align: center; } footer { padding: 0 15px 13px; color: #a9c0ae; font: 9px ui-monospace, monospace; }
  @media (max-width: 600px) { .table-zone { grid-template-columns: 60px 1fr 60px; gap: 6px; } .blind-tab { min-width: 125px; } .playing-card { width: 53px; height: 77px; } .playing-card > strong { inset: 25px 0 auto; font-size: 24px; } }
</style>
