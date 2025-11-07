package ma.emsi.test2;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaifAvecLogging {


    private static void configureLogger() {

        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); 

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {

        configureLogger();
        System.out.println("=== Logging configuré ===\n");

        String geminiApiKey = System.getenv("GEMINI");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement GEMINI n'est pas définie.");
            return;
        }

        System.out.println("=== PHASE 1 : Ingestion des documents ===");

        Path documentPath = Paths.get("src/main/resources/support_rag.pdf");
        System.out.println("Chargement du document : " + documentPath);

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        System.out.println("Document chargé avec succès");

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.printf("Document découpé en %d segments\n", segments.size());

        System.out.println("Création du modèle d'embedding...");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        System.out.println("Génération des embeddings...");
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.printf("%d embeddings créés\n", embeddings.size());

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Embeddings stockés en mémoire\n");

        System.out.println("=== PHASE 2 : Configuration de l'Assistant RAG ===");

        System.out.println("Connexion au modèle Gemini avec logging activé...");
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
        System.out.println("Récupérateur de contenu configuré");

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();
        System.out.println("Assistant RAG prêt avec logging activé !\n");

        Scanner scanner = new Scanner(System.in);
        System.out.println("Assistant RAG avec Logging - Tapez 'quitter' pour arrêter");
        System.out.println("Les détails des requêtes et réponses seront affichés dans la console\n");

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
                System.out.println("\n[Recherche et génération - Logs détaillés ci-dessous]");
                System.out.println("=".repeat(70));
                String reponse = assistant.chat(question);
                System.out.println("=".repeat(70));
                System.out.println("\nRéponse :");
                System.out.println(reponse);
                System.out.println();
            } catch (Exception e) {
                System.err.println("Erreur : " + e.getMessage());
                e.printStackTrace();
            }
        }
        scanner.close();
    }
}