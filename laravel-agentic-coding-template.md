---
layout: default
title: "Laravel Agentic Coding Template"
---

# Laravel Agentic Coding: Humans Design, AI Implements!

> This template adapts the Agentic Coding methodology for Laravel development. Customize this for your specific Laravel project (Laravel 10, 11, etc.). Throughout development, you should always (1) start with a small and simple solution, (2) design at a high level before implementation, and (3) frequently ask humans for feedback and clarification.
> {: .warning }

## Project Configuration

**Laravel Version:** [e.g., Laravel 11.x]
**PHP Version:** [e.g., PHP 8.2+]
**Database:** [e.g., MySQL 8.0, PostgreSQL 15]
**Frontend Stack:** [e.g., Blade + Alpine.js, Inertia.js + Vue 3, Livewire 3]
**Additional Packages:** [e.g., Spatie packages, Laravel Sanctum, etc.]

## Laravel Agentic Coding Steps

Agentic Coding should be a collaboration between Human System Design and AI Implementation:

| Steps             |   Human    |     AI     | Comment                                                                                        |
| :---------------- | :--------: | :--------: | :--------------------------------------------------------------------------------------------- |
| 1. Requirements   |  ★★★ High  |  ★☆☆ Low   | Humans understand the business requirements and Laravel context.                               |
| 2. Architecture   | ★★☆ Medium | ★★☆ Medium | Humans specify the high-level Laravel architecture, AI fills in implementation details.        |
| 3. Services       | ★★☆ Medium | ★★☆ Medium | Humans define external integrations and APIs, AI helps with Laravel service implementations.   |
| 4. Models & Logic |  ★☆☆ Low   |  ★★★ High  | AI designs Eloquent models, relationships, and business logic based on requirements.          |
| 5. Implementation |  ★☆☆ Low   |  ★★★ High  | AI implements controllers, views, routes, and middleware following Laravel conventions.        |
| 6. Optimization   | ★★☆ Medium | ★★☆ Medium | Humans evaluate results, AI helps optimize queries, caching, and performance.                 |
| 7. Testing        |  ★☆☆ Low   |  ★★★ High  | AI writes comprehensive tests (Feature, Unit, Browser) and handles edge cases.                |

### 1. Requirements

Clarify the requirements for your Laravel project and evaluate the best approach:

- **Laravel Strengths:**
  - **Good for**: CRUD applications, APIs, admin panels, e-commerce platforms
  - **Good for**: Rapid prototyping with Eloquent ORM and Blade templating
  - **Good for**: Authentication, authorization, and user management systems
  - **Not ideal for**: Real-time applications (consider Laravel WebSockets/Pusher), heavy computational tasks

- **Keep It User-Centric:** Focus on the end-user experience and business value
- **Laravel-First Approach:** Leverage Laravel's built-in features before adding complexity

**Example Requirements Template:**
```markdown
## User Stories
- As a [user type], I want to [action] so that [benefit]
- As a [user type], I want to [action] so that [benefit]

## Functional Requirements
- Authentication and authorization
- CRUD operations for [entities]
- API endpoints for [mobile app/frontend]
- Email notifications for [events]
- File uploads and management

## Non-Functional Requirements
- Performance: Page load < 2 seconds
- Security: OWASP compliance
- Scalability: Support [X] concurrent users
```

### 2. Architecture Design

Outline the Laravel application architecture and design patterns:

- **Identify Laravel Patterns:**
  - **Repository Pattern**: For complex data access logic
  - **Service Pattern**: For business logic separation
  - **Observer Pattern**: For model events and side effects
  - **Policy Pattern**: For authorization logic
  - **Job Pattern**: For background processing

- **Database Design:**
  ```mermaid
  erDiagram
      User ||--o{ Post : creates
      User {
          id bigint PK
          name string
          email string
          email_verified_at timestamp
          created_at timestamp
          updated_at timestamp
      }
      Post {
          id bigint PK
          user_id bigint FK
          title string
          content text
          published_at timestamp
          created_at timestamp
          updated_at timestamp
      }
  ```

- **Application Flow:**
  ```mermaid
  flowchart LR
      Request[HTTP Request] --> Middleware[Middleware Stack]
      Middleware --> Route[Route Resolution]
      Route --> Controller[Controller Action]
      Controller --> Service[Service Layer]
      Service --> Repository[Repository/Model]
      Repository --> Database[(Database)]
      Database --> Repository
      Repository --> Service
      Service --> Controller
      Controller --> View[View/JSON Response]
      View --> Response[HTTP Response]
  ```

### 3. Services & External Integrations

Based on the Architecture Design, identify and implement necessary services:

- **Laravel Services** (business logic layer):
  - Reading inputs (form requests, API calls, file uploads)
  - Writing outputs (email notifications, file generation, API responses)
  - External integrations (payment gateways, third-party APIs, cloud services)

- **Service Implementation Example:**
  ```php
  // app/Services/PaymentService.php
  <?php
  
  namespace App\Services;
  
  use App\Models\Order;
  use Illuminate\Support\Facades\Http;
  
  class PaymentService
  {
      public function processPayment(Order $order, array $paymentData): array
      {
          // Implementation here
          $response = Http::post('https://api.payment-provider.com/charge', [
              'amount' => $order->total,
              'currency' => 'USD',
              'source' => $paymentData['token'],
          ]);
          
          return $response->json();
      }
  }
  ```

- **Document Services:**
  - `name`: `PaymentService` (`app/Services/PaymentService.php`)
  - `input`: `Order $order, array $paymentData`
  - `output`: `array` (payment response)
  - `purpose`: Process payments through external gateway

### 4. Models & Database Design

Plan Eloquent models, relationships, and database structure:

- **Eloquent Models:**
  ```php
  // app/Models/User.php
  class User extends Authenticatable
  {
      protected $fillable = ['name', 'email', 'password'];
      
      public function posts(): HasMany
      {
          return $this->hasMany(Post::class);
      }
  }
  ```

- **Migration Design:**
  ```php
  // database/migrations/create_posts_table.php
  Schema::create('posts', function (Blueprint $table) {
      $table->id();
      $table->foreignId('user_id')->constrained()->onDelete('cascade');
      $table->string('title');
      $table->text('content');
      $table->timestamp('published_at')->nullable();
      $table->timestamps();
      
      $table->index(['user_id', 'published_at']);
  });
  ```

- **Model Relationships and Business Logic:**
  - Define relationships (hasMany, belongsTo, manyToMany)
  - Add accessors, mutators, and casts
  - Implement model events and observers
  - Add validation rules and form requests

### 5. Implementation

Implement Laravel components following conventions:

- **🎉 Agentic Implementation Begins!**
- **Follow Laravel Conventions**: Use Artisan commands, follow PSR standards
- **FAIL FAST**: Use Laravel's built-in validation and error handling
- **Leverage Laravel Features**: Use built-in authentication, authorization, caching, queues

**Implementation Checklist:**
- [ ] Routes (web.php, api.php)
- [ ] Controllers with proper HTTP methods
- [ ] Form Requests for validation
- [ ] Eloquent models with relationships
- [ ] Database migrations and seeders
- [ ] Blade views or API resources
- [ ] Middleware for authentication/authorization
- [ ] Service providers for dependency injection
- [ ] Configuration files

### 6. Optimization

- **Laravel-Specific Optimizations:**
  - **Database**: Query optimization, eager loading, database indexing
  - **Caching**: Route caching, config caching, view caching, Redis/Memcached
  - **Performance**: Queue jobs, horizon for monitoring, octane for speed
  - **Code**: Service container optimization, autoloader optimization

- **Optimization Examples:**
  ```php
  // Eager loading to prevent N+1 queries
  $users = User::with('posts.comments')->get();
  
  // Caching expensive operations
  $stats = Cache::remember('user-stats', 3600, function () {
      return User::selectRaw('COUNT(*) as total, AVG(age) as avg_age')->first();
  });
  
  // Background job processing
  SendWelcomeEmail::dispatch($user);
  ```

### 7. Testing & Reliability

- **Laravel Testing Strategy:**
  - **Feature Tests**: Test HTTP endpoints and user workflows
  - **Unit Tests**: Test individual classes and methods
  - **Browser Tests**: Test JavaScript interactions with Laravel Dusk
  - **API Tests**: Test API endpoints and responses

- **Testing Examples:**
  ```php
  // tests/Feature/PostTest.php
  public function test_user_can_create_post()
  {
      $user = User::factory()->create();
      
      $response = $this->actingAs($user)
          ->post('/posts', [
              'title' => 'Test Post',
              'content' => 'This is a test post content.'
          ]);
      
      $response->assertRedirect('/posts');
      $this->assertDatabaseHas('posts', [
          'title' => 'Test Post',
          'user_id' => $user->id
      ]);
  }
  ```

## Laravel Project File Structure

```
laravel-project/
├── app/
│   ├── Http/
│   │   ├── Controllers/
│   │   ├── Middleware/
│   │   └── Requests/
│   ├── Models/
│   ├── Services/
│   ├── Repositories/ (if using Repository pattern)
│   └── Policies/
├── database/
│   ├── migrations/
│   ├── seeders/
│   └── factories/
├── resources/
│   ├── views/
│   └── js/ (if using frontend build tools)
├── routes/
│   ├── web.php
│   ├── api.php
│   └── console.php
├── tests/
│   ├── Feature/
│   ├── Unit/
│   └── Browser/ (for Dusk tests)
├── config/
├── storage/
└── public/
```

## Laravel-Specific Best Practices

### Code Organization
- **Controllers**: Keep thin, delegate to services
- **Models**: Focus on relationships and data access
- **Services**: Handle business logic
- **Repositories**: Abstract data access (when needed)
- **Form Requests**: Handle validation logic
- **Resources**: Transform API responses

### Security
- Use Laravel's built-in CSRF protection
- Implement proper authorization with Gates and Policies
- Sanitize user input with validation rules
- Use Eloquent ORM to prevent SQL injection
- Implement rate limiting for APIs

### Performance
- Use database indexing strategically
- Implement caching at multiple levels
- Use queue jobs for heavy operations
- Optimize Eloquent queries (select only needed columns)
- Use Laravel Octane for production performance

### Testing
- Write tests for all critical business logic
- Use factories for test data generation
- Mock external services in tests
- Test both happy path and edge cases
- Maintain high test coverage

## Customization Notes

**For Laravel 10 Projects:**
- Update PHP version requirements (8.1+)
- Consider using Laravel Pennant for feature flags
- Leverage improved validation and routing features

**For Laravel 11 Projects:**
- Update PHP version requirements (8.2+)
- Use new application structure if applicable
- Leverage latest Eloquent and database features
- Consider new testing improvements

**Project-Specific Customizations:**
- [ ] Update Laravel version and PHP requirements
- [ ] Modify database configuration
- [ ] Add project-specific packages and dependencies
- [ ] Customize authentication and authorization requirements
- [ ] Add project-specific testing strategies
- [ ] Include deployment and environment configurations

---

*This template should be customized for each Laravel project. Remove sections that don't apply and add project-specific requirements, constraints, and conventions.*
