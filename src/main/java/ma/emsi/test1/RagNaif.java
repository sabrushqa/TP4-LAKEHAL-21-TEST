package ma.emsi.test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {
        String geminiApiKey = System.getenv("GEMINI");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement GEMINI n'est pas définie.");
            return;
        }

        System.out.println("=== PHASE 1 : Ingestion des documents ===");

        // 1. Chargement du document
        Path documentPath = Paths.get("src/main/resources/support_rag.pdf");
        System.out.println("Chargement du document : " + documentPath);

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        System.out.println("Document chargé avec succès");

        // 2. Découpage en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.printf("Document découpé en %d segments\n", segments.size());

        // 3. Création du modèle d'embedding
        System.out.println("Création du modèle d'embedding...");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 4. Génération des embeddings
        System.out.println("Génération des embeddings...");
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.printf("%d embeddings créés\n", embeddings.size());

        // 5. Stockage des embeddings
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Embeddings stockés en mémoire\n");

        System.out.println("=== PHASE 2 : Configuration de l'Assistant RAG ===");

        // 6. Création du modèle de chat (utiliser ChatModel dans la version 1.8.0)
        System.out.println("Connexion au modèle Gemini...");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.7)
                .build();

        // 7. Création du Content Retriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
        System.out.println("Récupérateur de contenu configuré");

        // 8. Création de l'assistant avec AiServices (utiliser .chatModel() dans la version 1.8.0)
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();
        System.out.println("Assistant RAG prêt !\n");

        // 9. Boucle de questions-réponses
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== Assistant RAG Naïf - Tapez 'quitter' pour arrêter ===\n");

        while (true) {
            System.out.print("Votre question : ");
            String question = scanner.nextLine().trim();

            if (question.equalsIgnoreCase("quitter") || question.equalsIgnoreCase("exit")) {
                System.out.println("\nAu revoir !");
                break;
            }

            if (question.isEmpty()) {
                System.out.println("Veuillez poser une question.\n");
                continue;
            }

            try {
                System.out.println("Recherche et génération de la réponse...");
                String reponse = assistant.chat(question);
                System.out.println("\n--- Réponse ---");
                System.out.println(reponse);
                System.out.println("---------------\n");
            } catch (Exception e) {
                System.err.println("Erreur lors du traitement : " + e.getMessage());
                e.printStackTrace();
            }
        }
        scanner.close();
    }
}