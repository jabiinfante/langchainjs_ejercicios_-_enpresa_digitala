/**
 * =============================================================================
 * EJERCICIO: Indexador de Documentación MDN en Vector Store (Qdrant)
 * =============================================================================
 *
 * Este script demuestra cómo crear un pipeline de indexación para RAG:
 * 1. Cargar documentos desde URLs (web scraping)
 * 2. Dividir documentos en chunks más pequeños
 * 3. Generar embeddings y almacenarlos en un vector store
 *
 * CONCEPTOS CLAVE:
 * - CheerioWebBaseLoader: Carga contenido HTML de URLs y extrae texto
 * - RecursiveCharacterTextSplitter: Divide documentos en chunks con overlap
 * - Embeddings: Representación vectorial del texto para búsqueda semántica
 * - Vector Store (Qdrant): Base de datos optimizada para búsqueda por similitud
 *
 * FLUJO DEL EJERCICIO:
 * 1. Definir URLs de MDN a indexar
 * 2. Cargar el contenido HTML de cada URL
 * 3. Limpiar y dividir en chunks
 * 4. **Eliminar vectores existentes** para evitar duplicados
 * 5. Generar embeddings e insertar en Qdrant
 * 6. Verificar con una búsqueda de prueba
 *
 * NOTA SOBRE DUPLICADOS:
 * Este script elimina los vectores existentes que coincidan con las URLs
 * a indexar antes de insertar los nuevos. Esto evita duplicados pero
 * re-indexa siempre el contenido.
 *
 * TODO: Lo ideal sería comprobar si el contenido ha cambiado (ej: hash del
 * contenido o fecha de modificación) antes de re-indexar, para evitar
 * trabajo innecesario. Por simplicidad, no lo implementamos aquí.
 * =============================================================================
 */

import { MistralAIEmbeddings } from "@langchain/mistralai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantClient } from "@qdrant/js-client-rest";

// =============================================================================
// PASO 1: Configuración
// =============================================================================

const COLLECTION_NAME = "langchainjs-testing-dia2";

// URLs de MDN a indexar (documentación sobre Web Storage)
const urls = [
  "https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API",
  "https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API/Using_the_Web_Storage_API",
  // Descomentar para indexar más documentación:
  // "https://developer.mozilla.org/en-US/docs/Web/API/Storage",
  // "https://developer.mozilla.org/en-US/docs/Web/API/Storage_API",
  // "https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria",
  // "https://developer.mozilla.org/en-US/docs/Web/API/StorageManager",
  // "https://developer.mozilla.org/en-US/docs/Web/API/Storage_Access_API",
  // "https://developer.mozilla.org/en-US/docs/Web/API/Storage_Access_API/Using",
  // "https://developer.mozilla.org/en-US/docs/Web/API/Shared_Storage_API",
];

// Configurar embeddings de Mistral
const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed",
});

// =============================================================================
// PASO 2: Cargar documentos desde las URLs
// =============================================================================
// CheerioWebBaseLoader usa Cheerio (parser HTML) para extraer contenido.
// El selector "main#content" extrae solo el contenido principal de MDN.

console.log("📥 Cargando documentos desde MDN...\n");

const loaders = urls.map(
  (url) => new CheerioWebBaseLoader(url, { selector: "main#content" }),
);

const docs = [];
for (const loader of loaders) {
  console.log(`  → ${loader.webPath}`);
  const loadedDocs = await loader.load();
  docs.push(...loadedDocs);
}

console.log(`\n✅ ${docs.length} documentos cargados\n`);

// =============================================================================
// PASO 3: Limpiar y dividir documentos en chunks
// =============================================================================
// - Limpiamos saltos de línea excesivos para reducir ruido
// - Dividimos en chunks de ~1000 caracteres con 150 de overlap
// - El overlap ayuda a mantener contexto entre chunks adyacentes

// Limpiar pageContent: reemplazar múltiples saltos de línea por uno solo
docs.forEach((doc) => {
  doc.pageContent = doc.pageContent.replace(/\n{2,}/g, "\n");
});

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,   // Tamaño máximo de cada chunk en caracteres
  chunkOverlap: 150, // Solapamiento entre chunks consecutivos
});

const allSplits = await splitter.splitDocuments(docs);
console.log(`📄 Documentos divididos en ${allSplits.length} chunks\n`);

// =============================================================================
// PASO 4: Eliminar vectores existentes para evitar duplicados
// =============================================================================
// Antes de insertar, eliminamos los vectores que ya existen para las URLs
// que vamos a indexar. Usamos el metadato "source" que contiene la URL.
//
// NOTA: Lo ideal sería verificar si el contenido ha cambiado antes de
// re-indexar (usando un hash o fecha de modificación), pero por simplicidad
// siempre eliminamos y re-insertamos.

console.log("🗑️  Eliminando vectores existentes para evitar duplicados...\n");

const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

// Eliminar vectores por cada URL (filtrando por el metadato "source")
for (const url of urls) {
  try {
    await qdrantClient.delete(COLLECTION_NAME, {
      filter: {
        must: [
          {
            key: "metadata.source",
            match: { value: url },
          },
        ],
      },
    });
    console.log(`  → Eliminados vectores de: ${url}`);
  } catch (error: any) {
    // Si la colección no existe, no hay nada que eliminar
    if (error.status === 404 || error.message?.includes("not found")) {
      console.log(`  → Colección no existe aún, se creará al insertar`);
      break; // No hace falta seguir intentando eliminar
    }
    throw error;
  }
}

console.log("");

// =============================================================================
// PASO 5: Generar embeddings e insertar en Qdrant
// =============================================================================
// QdrantVectorStore.fromDocuments():
// - Genera embeddings para cada chunk
// - Los inserta en la colección de Qdrant
// - Crea la colección si no existe

console.log("📤 Generando embeddings e insertando en Qdrant...\n");

const vectorStore = await QdrantVectorStore.fromDocuments(
  allSplits,
  embeddings,
  {
    url: process.env.QDRANT_URL,
    collectionName: COLLECTION_NAME,
    apiKey: process.env.QDRANT_API_KEY,
  },
);

console.log(`✅ ${allSplits.length} vectores insertados en "${COLLECTION_NAME}"\n`);

// =============================================================================
// PASO 6: Verificar con una búsqueda de prueba
// =============================================================================
// Hacemos una búsqueda semántica para verificar que todo funciona.

console.log("🔍 Verificando con búsqueda de prueba...\n");

const query = "¿Qué capacidad máxima en megas puedo usar para localStorage en un navegador?";
console.log(`Query: "${query}"\n`);

const retrievedDocs = await vectorStore.similaritySearch(query, 2);

console.log("Documentos recuperados:");
retrievedDocs.forEach((doc, idx) => {
  console.log(`\n--- Documento ${idx + 1} ---`);
  console.log(`Fuente: ${doc.metadata.source}`);
  console.log(`Contenido (primeros 300 chars):`);
  console.log(doc.pageContent.substring(0, 300) + "...");
});

console.log("\n✅ Indexación completada");
