import {Component, Inject} from '@angular/core';
import {ProjectService} from '../project.service';
import {Paginator} from "../paginator";
import {Project, ProjectFilter} from "../models";
import {Location} from '@angular/common';
import {MAT_DIALOG_DATA, MatDialog, MatDialogRef, MatIconRegistry} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";
import {DialogComponent} from "../dialog/dialog.component";

@Component({
    selector: 'app-project',
    templateUrl: './project.component.html',
    styleUrls: ['./project.component.css']
})
export class ProjectComponent extends Paginator<Project> {

    displayed_columns: string[] = ['name', 'dag_count', 'last_activity', 'links'];
    name: string;

    constructor(protected project_service: ProjectService, protected location: Location,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                public dialog: MatDialog
    ) {
        super(project_service, location);
        iconRegistry.addSvgIcon('remove',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/trash.svg'));
    }

    get_filter() {
        let res = new ProjectFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        return res;
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }


    remove(element: Project) {
        const dialogRef = this.dialog.open(DialogComponent, {
            width: '550px', height: '200px',
            data: {'message': 'The all content will be deleted. Do you want to continue?'}
        });

        dialogRef.afterClosed().subscribe(result => {
            if(result){
                this.project_service.remove(element.id).subscribe(data=>{
                    this.change.emit();
                });
            }
        });

    }
}
