import { NgModule }       from '@angular/core';
import { BrowserModule }  from '@angular/platform-browser';
import { FormsModule }    from '@angular/forms';
import { HttpClientModule }    from '@angular/common/http';

// import { HttpClientInMemoryWebApiModule } from 'angular-in-memory-web-api';
// import { InMemoryDataService }  from './in-memory-data.service';

import { AppRoutingModule }     from './app-routing.module';

import { AppComponent }         from './app.component';
import { ProjectComponent }   from   './project/project.component';
import { TaskDetailComponent }  from './task-detail/task-detail.component';
import { TaskComponent }      from './task/task.component';
import { MessagesComponent }    from './messages/messages.component';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {DemoMaterialModule} from './material-module';
import { DagComponent } from './dag/dag.component';
import { DagDetailComponent } from './dag-detail/dag-detail.component';
import { ComputerComponent } from './computer/computer.component';

@NgModule({
  imports: [
    BrowserModule,
    FormsModule,
    AppRoutingModule,
    HttpClientModule,
    BrowserAnimationsModule,
    DemoMaterialModule,

    // The HttpClientInMemoryWebApiModule module intercepts HTTP requests
    // and returns simulated server responses.
    // Remove it when a real server is ready to receive requests.
    // HttpClientInMemoryWebApiModule.forRoot(
    //   InMemoryDataService, { dataEncapsulation: false }
    // )
  ],
  declarations: [
    AppComponent,
    ProjectComponent,
    TaskDetailComponent,
    TaskComponent,
    MessagesComponent,
    DagComponent,
    DagDetailComponent,
    ComputerComponent
  ],
  bootstrap: [ AppComponent ]
})
export class AppModule { }
