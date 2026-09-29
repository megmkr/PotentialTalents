import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

const candidateList = document.querySelector("#candidate-list");
const searchInput = document.querySelector("#search-input");
const resultCount = document.querySelector("#result-count");

const candidates = await fetch("data/processed/candidates.json").then((response) => response.json());

const starred = new Set();

function cosineSimilarity(a, b) {
   let dot = 0;
   let magnitudeA = 0;
   let magnitudeB = 0;

   for (let index = 0; index < a.length; index += 1) {
      dot += a[index] * b[index];
      magnitudeA += a[index] ** 2;
      magnitudeB += b[index] ** 2;
   }

   return dot / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
}
function connectionCount(candidate) {
   const value = candidate.connection_clean ?? candidate.connections ?? 0;
   const number = Number(String(value).replace(/[^0-9]/g, ""));
   return Number.isNaN(number) ? 0 : number;
}

async function rankedCandidates(query) {
   const output = await extractor(query, {
      pooling: "mean",
      normalize: true,
   });

   const queryVector = Array.from(output.data);

   return candidates
      .map((candidate, index) => ({
         ...candidate,
         index,
         score: cosineSimilarity(queryVector, candidate.vector),
         starred: starred.has(index),
      }))
      .sort((a, b) => {
         if (a.starred !== b.starred) {
            return Number(b.starred) - Number(a.starred);
         }

         const similarityDifference = b.score - a.score;

         if (similarityDifference !== 0) {
            return similarityDifference;
         }

         return connectionCount(b) - connectionCount(a);
      });
}

function renderCandidates(results) {
   resultCount.textContent = `${results.length} candidates · ${starred.size} starred`;

   candidateList.innerHTML = results
      .map(
         (candidate) => `
        <article class="candidate">
          <div>
            <h3>${candidate.job_title_clean ?? ""}</h3>
            <p>${candidate.location_clean ?? ""}</p>
            <small>
                     Connections:
                     ${candidate.connection_clean}
            </small>
            </br>
            <small>Similarity: ${candidate.score.toFixed(3)}</small>
          </div>
          <button
            class="star-button"
            data-index="${candidate.index}"
            aria-label="Star candidate"
          >
            ${candidate.starred ? "★" : "☆"}
          </button>
        </article>
      `,
      )
      .join("");
}

async function search() {
   const query = searchInput.value.trim();

   if (!query) {
      renderCandidates([]);
      return;
   }

   resultCount.textContent = "Ranking candidates...";
   const results = await rankedCandidates(query);
   renderCandidates(results);
}

searchInput.addEventListener("input", search);

candidateList.addEventListener("click", async (event) => {
   const button = event.target.closest(".star-button");

   if (!button) {
      return;
   }

   const index = Number(button.dataset.index);

   if (starred.has(index)) {
      starred.delete(index);
   } else {
      starred.add(index);
   }

   await search();
});

await search();
