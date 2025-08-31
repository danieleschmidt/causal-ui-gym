// Enhanced Error Handling Module
// Generated: 2025-08-31T03:17:13.205Z

class ApplicationError extends Error {
    constructor(message, statusCode = 500, context = {}) {
        super(message);
        this.name = this.constructor.name;
        this.statusCode = statusCode;
        this.context = context;
        this.timestamp = new Date().toISOString();
        Error.captureStackTrace(this, this.constructor);
    }
}

class ValidationError extends ApplicationError {
    constructor(message, field, value) {
        super(message, 400, { field, value });
    }
}

class AuthenticationError extends ApplicationError {
    constructor(message = 'Authentication failed') {
        super(message, 401);
    }
}

class AuthorizationError extends ApplicationError {
    constructor(message = 'Access denied') {
        super(message, 403);
    }
}

class NotFoundError extends ApplicationError {
    constructor(resource, identifier) {
        super(`${resource} not found: ${identifier}`, 404, { resource, identifier });
    }
}

class RateLimitError extends ApplicationError {
    constructor(limit, window) {
        super(`Rate limit exceeded: ${limit} requests per ${window}`, 429, { limit, window });
    }
}

// Global error handler
function errorHandler(err, req, res, next) {
    const error = err instanceof ApplicationError ? err : new ApplicationError(err.message);
    
    // Log error
    console.error({
        timestamp: error.timestamp,
        error: error.message,
        statusCode: error.statusCode,
        context: error.context,
        stack: error.stack
    });
    
    // Send response
    res.status(error.statusCode).json({
        error: {
            message: error.message,
            statusCode: error.statusCode,
            timestamp: error.timestamp,
            ...(process.env.NODE_ENV === 'development' && { stack: error.stack })
        }
    });
}

module.exports = {
    ApplicationError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    errorHandler
};
