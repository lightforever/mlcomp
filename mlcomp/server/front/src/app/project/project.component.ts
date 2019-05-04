import {Component} from '@angular/core';
import {ProjectService} from '../project.service';
import {Paginator} from "../paginator";
import {Project, ProjectFilter} from "../models";
import {Location} from '@angular/common';

@Component({
    selector: 'app-project',
    templateUrl: './project.component.html',
    styleUrls: ['./project.component.css']
})
export class ProjectComponent extends Paginator<Project>{

    displayed_columns: string[] = ['name', 'dag_count', 'last_activity'];
    name: string;

    constructor(protected project_service: ProjectService, protected location:Location) {
        super(project_service, location)
    }

    get_filter(){
        let res = new ProjectFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        return res;
    }

    filter_name(name: string){
        this.name = name;
        this.change.emit();
    }


}
