import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { routes } from './routes'; // Import the routes array
// Import components if needed

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
