import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';
import {FormsModule} from '@angular/forms';
import {HttpClientModule} from '@angular/common/http';
import {MatPaginatorModule} from '@angular/material/paginator';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {DemoMaterialModule} from './material-module';

import {AppRoutingModule} from './app-routing.module';

import {AppComponent} from './app.component';
import {ProjectComponent} from './project/project.component';
import {TaskDetailComponent} from './task-detail/task-detail.component';
import {TaskComponent} from './task/task.component';
import {ComputerComponent} from './computer/computer.component';
import {MessagesComponent} from './messages/messages.component';
import {DagModule} from "./dag/dag.module";

@NgModule({
    imports: [
        BrowserModule,
        FormsModule,
        AppRoutingModule,
        HttpClientModule,
        BrowserAnimationsModule,
        DemoMaterialModule,
        MatPaginatorModule,
        DagModule
    ],
    declarations: [
        AppComponent,
        ProjectComponent,
        TaskDetailComponent,
        TaskComponent,
        MessagesComponent,
        ComputerComponent
    ],
    bootstrap: [AppComponent]
})
export class AppModule {
}
