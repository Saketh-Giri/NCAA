const CSV_PATH = "../outputs/official_bracket_predictions.csv";

const REGION_CODES = ["W", "X", "Y", "Z"];
const REGION_LABELS = {
  W: "Region W",
  X: "Region X",
  Y: "Region Y",
  Z: "Region Z",
};
const ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"];
const SEED_PAIRINGS = [
  [1, 16],
  [8, 9],
  [5, 12],
  [4, 13],
  [6, 11],
  [3, 14],
  [7, 10],
  [2, 15],
];

let tournamentRows = [];
let selectedYear = null;
let bracket = [];
let finalFour = {
  semifinals: [
    [null, null],
    [null, null],
  ],
  final: [null, null],
  champion: null,
};
let showActualWins = false;

const bracketEl = document.getElementById("bracket");
const statusEl = document.getElementById("status");
const championNameEl = document.getElementById("championName");
const championMetaEl = document.getElementById("championMeta");
const teamTemplate = document.getElementById("teamButtonTemplate");
const yearSelectEl = document.getElementById("yearSelect");

document.getElementById("resetBtn").addEventListener("click", () => {
  resetSelections();
  render();
});

document.getElementById("autoPickBtn").addEventListener("click", () => {
  autoPickBracket();
  render();
});

yearSelectEl.addEventListener("change", (event) => {
  selectedYear = Number(event.target.value);
  resetSelections();
  render();
});

document.getElementById("showActualToggle").addEventListener("change", (event) => {
  showActualWins = event.target.checked;
  render();
});

loadTeams();

async function loadTeams() {
  setStatus("Loading official bracket predictions...");

  try {
    const response = await fetch(CSV_PATH);
    if (!response.ok) {
      throw new Error(`Could not load ${CSV_PATH}`);
    }

    const csv = await response.text();
    tournamentRows = parseCsv(csv)
      .map((row) => ({
        team: row.TEAM,
        conference: row.CONF,
        seed: Number(row.SEED_NUM),
        bracketSeed: row.BRACKET_SEED,
        slotSeed: row.SLOT_SEED,
        regionCode: row.REGION_CODE,
        year: Number(row.YEAR),
        predictedWins: Number(row.Predicted_Tour_Wins),
        actualWins: Number(row.Tour_Wins),
        postseason: row.POSTSEASON,
      }))
      .filter(
        (team) =>
          Number.isFinite(team.seed) &&
          Number.isFinite(team.year) &&
          Number.isFinite(team.predictedWins),
      )
      .sort(
        (a, b) =>
          b.year - a.year ||
          a.regionCode.localeCompare(b.regionCode) ||
          a.seed - b.seed ||
          a.bracketSeed.localeCompare(b.bracketSeed),
      );

    populateYearSelect();
    selectedYear = Number(yearSelectEl.value);
    resetSelections();
    clearStatus();
    render();
  } catch (error) {
    setStatus(
      `Unable to load predictions. Start a local server from the project root, for example: python3 -m http.server 8000. Details: ${error.message}`,
      true,
    );
  }
}

function populateYearSelect() {
  const years = [...new Set(tournamentRows.map((team) => team.year))].sort((a, b) => b - a);
  yearSelectEl.innerHTML = "";

  for (const year of years) {
    const option = document.createElement("option");
    option.value = year;
    option.textContent = year;
    yearSelectEl.appendChild(option);
  }
}

function parseCsv(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  const headers = splitCsvLine(lines[0]);

  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
}

function buildOfficialBracket(year) {
  const teamsForYear = tournamentRows.filter((team) => team.year === year);

  return REGION_CODES.map((regionCode) => {
    const roundOf64 = SEED_PAIRINGS.map(([leftSeed, rightSeed]) => [
      resolveSeedSlot(teamsForYear, regionCode, leftSeed),
      resolveSeedSlot(teamsForYear, regionCode, rightSeed),
    ]);

    return {
      name: REGION_LABELS[regionCode] ?? `Region ${regionCode}`,
      rounds: [
        roundOf64,
        Array.from({ length: 4 }, () => [null, null]),
        Array.from({ length: 2 }, () => [null, null]),
        [[null, null]],
      ],
      champion: null,
    };
  });
}

function resolveSeedSlot(teamsForYear, regionCode, seed) {
  const slotSeed = `${regionCode}${String(seed).padStart(2, "0")}`;
  const candidates = teamsForYear.filter((team) => team.slotSeed === slotSeed);

  if (candidates.length <= 1) {
    return candidates[0] ?? null;
  }

  const winner = pickByModel(candidates);
  return {
    ...winner,
    playInTeams: candidates.map((team) => team.team),
  };
}

function resetSelections() {
  bracket = buildOfficialBracket(selectedYear);
  finalFour = {
    semifinals: [
      [null, null],
      [null, null],
    ],
    final: [null, null],
    champion: null,
  };
}

function autoPickBracket() {
  resetSelections();
  fillBracket(pickByModel);
}

function fillBracket(pickWinner) {
  for (let regionIndex = 0; regionIndex < bracket.length; regionIndex += 1) {
    for (let roundIndex = 0; roundIndex < 4; roundIndex += 1) {
      const matches = bracket[regionIndex].rounds[roundIndex];
      for (let matchIndex = 0; matchIndex < matches.length; matchIndex += 1) {
        const winner = pickWinner(matches[matchIndex]);
        if (winner) {
          advanceRegionWinner(regionIndex, roundIndex, matchIndex, winner);
        }
      }
    }
  }

  for (let semiIndex = 0; semiIndex < finalFour.semifinals.length; semiIndex += 1) {
    const winner = pickWinner(finalFour.semifinals[semiIndex]);
    if (winner) {
      advanceFinalFourWinner(semiIndex, winner);
    }
  }

  const champion = pickWinner(finalFour.final);
  if (champion) {
    finalFour.champion = champion;
  }
}

function pickByModel(match) {
  const available = match.filter(Boolean);
  if (available.length === 0) {
    return null;
  }

  return [...available].sort(
    (a, b) =>
      b.predictedWins - a.predictedWins ||
      a.seed - b.seed ||
      a.team.localeCompare(b.team),
  )[0];
}

function render() {
  bracketEl.innerHTML = "";

  for (let regionIndex = 0; regionIndex < bracket.length; regionIndex += 1) {
    bracketEl.appendChild(renderRegion(bracket[regionIndex], regionIndex));
  }

  const finalFourEl = renderFinalFour();
  bracketEl.insertAdjacentElement("afterend", finalFourEl);
  updateChampion();
}

function renderRegion(region, regionIndex) {
  const section = document.createElement("section");
  section.className = "region";

  const heading = document.createElement("h2");
  heading.textContent = region.name;
  section.appendChild(heading);

  const roundsEl = document.createElement("div");
  roundsEl.className = "rounds";

  region.rounds.forEach((round, roundIndex) => {
    const roundEl = document.createElement("div");
    roundEl.className = "round";

    const roundHeading = document.createElement("h3");
    roundHeading.textContent = ROUND_NAMES[roundIndex];
    roundEl.appendChild(roundHeading);

    round.forEach((match, matchIndex) => {
      roundEl.appendChild(
        renderMatch(match, (team) => {
          advanceRegionWinner(regionIndex, roundIndex, matchIndex, team);
          render();
        }),
      );
    });

    roundsEl.appendChild(roundEl);
  });

  section.appendChild(roundsEl);
  return section;
}

function renderFinalFour() {
  document.querySelector(".final-four")?.remove();

  const section = document.createElement("section");
  section.className = "final-four";

  const heading = document.createElement("h2");
  heading.textContent = "Final Four";
  section.appendChild(heading);

  finalFour.semifinals.forEach((match, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "round";

    const headingEl = document.createElement("h3");
    headingEl.textContent = `Semifinal ${index + 1}`;
    wrapper.appendChild(headingEl);
    wrapper.appendChild(
      renderMatch(match, (team) => {
        advanceFinalFourWinner(index, team);
        render();
      }),
    );
    section.appendChild(wrapper);
  });

  const finalWrapper = document.createElement("div");
  finalWrapper.className = "round";
  const finalHeading = document.createElement("h3");
  finalHeading.textContent = "Championship";
  finalWrapper.appendChild(finalHeading);
  finalWrapper.appendChild(
    renderMatch(finalFour.final, (team) => {
      finalFour.champion = team;
      render();
    }),
  );
  section.appendChild(finalWrapper);

  return section;
}

function renderMatch(match, onPick) {
  const matchEl = document.createElement("div");
  matchEl.className = `match${match.every((team) => !team) ? " is-empty" : ""}`;

  for (let index = 0; index < 2; index += 1) {
    matchEl.appendChild(renderTeamButton(match[index], onPick));
  }

  return matchEl;
}

function renderTeamButton(team, onPick) {
  const button = teamTemplate.content.firstElementChild.cloneNode(true);

  if (!team) {
    button.classList.add("is-placeholder");
    button.disabled = true;
    button.querySelector(".seed").textContent = "-";
    button.querySelector(".team-name").textContent = "TBD";
    button.querySelector(".team-meta").textContent = "Pick earlier winners";
    button.querySelector(".prediction").textContent = "";
    return button;
  }

  button.querySelector(".seed").textContent = team.seed;
  button.querySelector(".team-name").textContent = team.team;
  const playInLabel = team.playInTeams
    ? ` | First Four pick from ${team.playInTeams.join(" / ")}`
    : "";
  button.querySelector(
    ".team-meta",
  ).textContent = `${team.year} | ${team.bracketSeed} | ${team.conference} | ${team.postseason}${playInLabel}`;
  button.querySelector(".prediction").textContent = showActualWins
    ? `${team.actualWins} actual`
    : `${team.predictedWins.toFixed(2)} pred`;
  button.addEventListener("click", () => onPick(team));

  if (team.selected || finalFour.champion?.team === team.team) {
    button.classList.add("is-selected");
  }

  return button;
}

function advanceRegionWinner(regionIndex, roundIndex, matchIndex, team) {
  const region = bracket[regionIndex];
  region.rounds[roundIndex][matchIndex] = markSelection(
    region.rounds[roundIndex][matchIndex],
    team,
  );

  if (roundIndex < 3) {
    const nextMatchIndex = Math.floor(matchIndex / 2);
    const nextSlot = matchIndex % 2;
    const nextRoundIndex = roundIndex + 1;
    const nextMatch = clearSelection(region.rounds[nextRoundIndex][nextMatchIndex]);

    clearRegionAfter(region, nextRoundIndex, nextMatchIndex);
    nextMatch[nextSlot] = { ...team, selected: false };
    region.rounds[nextRoundIndex][nextMatchIndex] = nextMatch;
    region.champion = null;
  } else {
    region.champion = { ...team, selected: false };
  }

  syncFinalFourFromRegions();
}

function markSelection(match, selectedTeam) {
  return match.map((team) => team && { ...team, selected: team.team === selectedTeam.team });
}

function clearSelection(match) {
  return match.map((team) => team && { ...team, selected: false });
}

function clearRegionAfter(region, roundIndex, matchIndex) {
  let affectedMatchIndex = matchIndex;

  for (let nextRound = roundIndex + 1; nextRound < region.rounds.length; nextRound += 1) {
    affectedMatchIndex = Math.floor(affectedMatchIndex / 2);
    const match = region.rounds[nextRound][affectedMatchIndex];
    if (!match) {
      break;
    }

    match[0] = null;
    match[1] = null;
  }
}

function syncFinalFourFromRegions() {
  finalFour.semifinals = [
    [bracket[0]?.champion ?? null, bracket[1]?.champion ?? null],
    [bracket[2]?.champion ?? null, bracket[3]?.champion ?? null],
  ];
  finalFour.final = [null, null];
  finalFour.champion = null;
}

function advanceFinalFourWinner(semifinalIndex, team) {
  finalFour.semifinals[semifinalIndex] = markSelection(
    finalFour.semifinals[semifinalIndex],
    team,
  );
  finalFour.final[semifinalIndex] = team;
  finalFour.champion = null;
}

function updateChampion() {
  const champion = finalFour.champion;

  if (!champion) {
    championNameEl.textContent = "Choose winners to fill the bracket";
    championMetaEl.textContent = "";
    return;
  }

  championNameEl.textContent = champion.team;
  championMetaEl.textContent = `${champion.year} | ${champion.bracketSeed} | ${champion.conference} | ${champion.predictedWins.toFixed(
    2,
  )} predicted wins | ${champion.actualWins} actual wins`;
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.add("is-visible");
  statusEl.style.color = isError ? "#b42318" : "";
}

function clearStatus() {
  statusEl.textContent = "";
  statusEl.classList.remove("is-visible");
  statusEl.style.color = "";
}
